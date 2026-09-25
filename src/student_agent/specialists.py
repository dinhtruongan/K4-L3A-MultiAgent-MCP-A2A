"""Specialist agents. Each one owns a fixed set of MCP tool domains (least privilege)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from .contracts import ContractError
from .evidence import (
    pick,
    pick_shallow,
    records,
    strings,
    to_datetime,
    to_float,
    unique,
)
from .mcp_gateway import EvidenceGateway, MCPToolError, ToolSpec
from .state import AgentMessage, CaseState, Evidence, Finding
from .trace import TraceWriter

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (0.5, 1.5)
MAX_ENTITY_CALLS = 5

# Accepted names for each argument a tool may declare. Values come only from the case
# input or from earlier evidence; nothing is invented.
ARGUMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "order_id": ("order_id",),
    "customer_id": ("customer_id",),
    "seller_id": ("seller_id",),
    "product_id": ("product_id",),
    "item_id": ("item_id", "order_item_id"),
    "payment_id": ("payment_reference", "payment_id"),
    "payment_reference": ("payment_reference", "payment_id"),
    "shipment_id": ("shipment_id", "tracking_id"),
    "tracking_id": ("tracking_id", "shipment_id"),
    "refund_id": ("refund_id",),
    "policy_id": ("policy_id", "policy_version"),
    "policy_version": ("policy_version", "policy_id"),
}


class ToolUnavailable(RuntimeError):
    pass


def _argument_value(prop: str, context: dict[str, Any]) -> Any:
    for alias in ARGUMENT_ALIASES.get(prop, (prop,)):
        value = context.get(alias)
        if value not in (None, ""):
            return value
    return None


def build_arguments(spec: ToolSpec, context: dict[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for prop in spec.properties:
        if prop == "case_id":
            continue
        value = _argument_value(prop, context)
        if value is not None:
            arguments[prop] = value
        elif prop in spec.required:
            raise ToolUnavailable(f"missing_argument:{spec.name}.{prop}")
    return arguments


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, MCPToolError):
        return not (exc.not_found or exc.forbidden)
    return not isinstance(exc, ContractError | ValueError | ToolUnavailable)


class Specialist:
    name = "specialist"
    # domain -> candidate tool names, in order of preference
    tools: dict[str, tuple[str, ...]] = {}

    def __init__(
        self, gateway: EvidenceGateway, trace: TraceWriter, specs: dict[str, ToolSpec]
    ) -> None:
        self.gateway = gateway
        self.trace = trace
        self.specs = specs

    def tool_for(self, domain: str) -> ToolSpec | None:
        for candidate in self.tools.get(domain, ()):
            if candidate in self.specs:
                return self.specs[candidate]
        return None

    async def fetch(
        self,
        state: CaseState,
        finding: Finding,
        domain: str,
        context: dict[str, Any],
        purpose: str,
    ) -> Evidence | None:
        spec = self.tool_for(domain)
        if spec is None:
            finding.errors.append(f"tool_not_discovered:{domain}")
            return None
        try:
            arguments = build_arguments(spec, context)
        except ToolUnavailable as exc:
            finding.errors.append(str(exc))
            return None
        envelope: dict[str, Any] | None = None
        error: BaseException | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                envelope = await self.gateway.call(spec.name, case_id=state.case_id, **arguments)
                break
            except Exception as exc:  # noqa: BLE001 - classified by _is_transient
                error = exc
                if attempt == MAX_ATTEMPTS or not _is_transient(exc):
                    break
                await asyncio.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
        if envelope is None:
            if isinstance(error, MCPToolError) and error.not_found:
                code = "not_found"
            elif isinstance(error, MCPToolError) and error.forbidden:
                code = "forbidden"
            elif isinstance(error, ContractError | ValueError):
                code = "invalid_envelope"
            else:
                code = "tool_error"
            finding.errors.append(f"{code}:{spec.name}")
            return None
        evidence = Evidence(
            ref=envelope["evidence_ref"],
            tool=spec.name,
            domain=envelope["domain"],
            data=envelope["data"],
            actor=self.name,
            warnings=tuple(envelope.get("warnings") or ()),
        )
        finding.evidence.append(evidence)
        self.trace.emit(
            case_id=state.case_id,
            event_type="tool_result_consumed",
            actor=self.name,
            tool_name=spec.name,
            evidence_refs=[evidence.ref],
            attributes={
                "domain": evidence.domain,
                "purpose": purpose,
                "warnings": len(evidence.warnings),
            },
        )
        return evidence

    async def run(self, state: CaseState, message: AgentMessage) -> Finding:
        raise NotImplementedError


def _iso(value: Any) -> str | None:
    parsed = to_datetime(value)
    return parsed.isoformat() if parsed else None


def _item_record(raw: dict[str, Any]) -> dict[str, Any]:
    price = to_float(pick_shallow(raw, "price", "item_price", "unit_price", "amount"))
    freight = to_float(pick_shallow(raw, "freight_value", "freight", "shipping_cost"))
    item_id = pick_shallow(raw, "item_id", "order_item_ref", "order_item_key", "id")
    sequence = pick_shallow(raw, "order_item_id", "item_sequence", "sequence")
    return {
        "item_id": str(item_id) if item_id is not None else None,
        "order_item_id": sequence,
        "product_id": pick_shallow(raw, "product_id"),
        "seller_id": pick_shallow(raw, "seller_id"),
        "price": price,
        "freight": freight,
        "shipping_limit": _iso(pick_shallow(raw, "shipping_limit_date", "shipping_limit")),
        "status": pick_shallow(raw, "status", "item_status"),
    }


class OrderItemAgent(Specialist):
    name = "order-agent"
    tools = {
        "order": ("get_order", "get_order_details"),
        "item": ("get_order_items", "get_items", "get_item", "list_order_items"),
        "seller": ("get_seller", "get_seller_profile"),
    }

    async def run(self, state: CaseState, message: AgentMessage) -> Finding:
        finding = Finding(agent=self.name)
        hints = state.hints
        order = await self.fetch(state, finding, "order", hints, "order_status_and_timeline")
        data = order.data if order else {}
        order_id = pick(data, "order_id") or hints.get("order_id")
        facts: dict[str, Any] = {
            "order_found": order is not None,
            "order_id": order_id,
            "order_status": (str(pick(data, "order_status", "status") or "").lower() or None),
            "customer_id": pick(data, "customer_id"),
            "purchase_ts": _iso(pick(data, "order_purchase_timestamp", "purchase_timestamp")),
            "approved_ts": _iso(pick(data, "order_approved_at", "approved_at")),
            "delivered_carrier_ts": _iso(
                pick(data, "order_delivered_carrier_date", "delivered_carrier_date")
            ),
            "delivered_customer_ts": _iso(
                pick(data, "order_delivered_customer_date", "delivered_customer_date")
            ),
            "estimated_ts": _iso(
                pick(data, "order_estimated_delivery_date", "estimated_delivery_date")
            ),
            "order_total": to_float(pick(data, "order_total", "total_amount", "total_value")),
            "canceled_by": pick(data, "canceled_by", "cancelled_by", "cancellation_actor"),
            "cancel_reason": pick(data, "cancellation_reason", "cancel_reason"),
            "shipment_ids": strings([pick(data, "shipment_id")]),
            "payment_references": strings([pick(data, "payment_reference", "payment_id")]),
        }
        raw_items = records(pick_shallow(data, "items", "order_items") or [], marker="price")
        context = {**hints, "order_id": order_id, "customer_id": facts["customer_id"]}
        if not raw_items:
            item_evidence = await self.fetch(state, finding, "item", context, "item_prices")
            if item_evidence:
                raw_items = records(item_evidence.data, "items", "order_items", marker="price")
        items = [_item_record(raw) for raw in raw_items]
        facts["items"] = items
        if facts["order_total"] is None and items and all(i["price"] is not None for i in items):
            facts["order_total"] = round(
                sum((i["price"] or 0) + (i["freight"] or 0) for i in items), 2
            )
        facts["freight_total"] = round(sum(i["freight"] or 0 for i in items), 2)
        facts["seller_ids"] = strings(
            [*(i["seller_id"] for i in items), pick(data, "seller_id"), hints.get("seller_id")]
        )
        limits = [to_datetime(i["shipping_limit"]) for i in items if i["shipping_limit"]]
        facts["shipping_limit_ts"] = max(limits).isoformat() if limits else None

        sellers: dict[str, dict[str, Any]] = {}
        if message.payload.get("need_seller", True):
            for seller_id in facts["seller_ids"][:MAX_ENTITY_CALLS]:
                evidence = await self.fetch(
                    state, finding, "seller", {**context, "seller_id": seller_id}, "seller_profile"
                )
                if evidence:
                    sellers[seller_id] = {
                        "ref": evidence.ref,
                        "state": pick(evidence.data, "seller_state", "state"),
                    }
        facts["sellers"] = sellers
        finding.facts = facts
        return finding


class PaymentAgent(Specialist):
    name = "payment-agent"
    tools = {
        "payment": ("get_payment", "get_payments", "get_order_payments"),
        "refund": ("get_refund", "get_refunds", "get_refund_status"),
    }

    async def run(self, state: CaseState, message: AgentMessage) -> Finding:
        finding = Finding(agent=self.name)
        context = {**state.hints, **message.payload.get("context", {})}
        payment = await self.fetch(state, finding, "payment", context, "payment_ledger")
        payments: list[dict[str, Any]] = []
        if payment:
            for raw in records(payment.data, "payments", "transactions", marker="payment_value"):
                value = to_float(pick_shallow(raw, "payment_value", "value", "amount"))
                if value is None:
                    continue
                payments.append(
                    {
                        "reference": pick_shallow(
                            raw, "payment_reference", "payment_id", "transaction_id", "reference"
                        ),
                        "sequential": pick_shallow(raw, "payment_sequential", "sequential"),
                        "type": pick_shallow(raw, "payment_type", "type", "method"),
                        "installments": pick_shallow(raw, "payment_installments", "installments"),
                        "value": value,
                        "status": str(pick_shallow(raw, "status", "payment_status") or "").lower()
                        or None,
                        "captured_at": _iso(
                            pick_shallow(raw, "captured_at", "paid_at", "created_at", "timestamp")
                        ),
                    }
                )
        refunds: list[dict[str, Any]] = []
        if payment:
            refunds.extend(_refunds_from(pick_shallow(payment.data, "refunds", "refund")))
        if message.payload.get("need_refund") and self.tool_for("refund") is not None:
            refund = await self.fetch(state, finding, "refund", context, "refund_status")
            if refund:
                refunds.extend(_refunds_from(refund.data))
        active = [p for p in payments if p["status"] not in {"failed", "declined", "voided"}]
        finding.facts = {
            "payment_found": payment is not None,
            "payments": payments,
            "paid_total": round(sum(p["value"] for p in active), 2) if payment else None,
            "payment_count": len(active),
            "payment_types": unique(p["type"] for p in active),
            "payment_references": strings(p["reference"] for p in payments),
            "refunds": refunds,
            "payment_total_field": to_float(
                pick(payment.data, "total_paid", "paid_total", "payment_total")
            )
            if payment
            else None,
        }
        return finding


def _refunds_from(data: Any) -> Iterable[dict[str, Any]]:
    if not data:
        return []
    result = []
    for raw in records(data, "refunds", marker="refund_status"):
        status = pick_shallow(raw, "refund_status", "status", "state")
        amount = to_float(pick_shallow(raw, "amount", "refund_amount", "value", "amount_brl"))
        if status is None and amount is None:
            continue
        result.append(
            {
                "refund_id": pick_shallow(raw, "refund_id", "id", "reference"),
                "status": str(status or "").lower() or None,
                "amount": amount,
                "reason": pick_shallow(raw, "reason", "failure_reason", "reason_code"),
                "requested_at": _iso(pick_shallow(raw, "requested_at", "created_at")),
            }
        )
    return result


class ShipmentAgent(Specialist):
    name = "shipment-agent"
    tools = {"shipment": ("get_shipment", "get_shipments", "get_tracking", "get_delivery")}

    async def run(self, state: CaseState, message: AgentMessage) -> Finding:
        finding = Finding(agent=self.name)
        context = {**state.hints, **message.payload.get("context", {})}
        shipment = await self.fetch(state, finding, "shipment", context, "delivery_timeline")
        data = shipment.data if shipment else {}
        finding.facts = {
            "shipment_found": shipment is not None,
            "shipment_ids": strings([pick(data, "shipment_id", "tracking_id", "tracking_code")]),
            "carrier": pick(data, "carrier", "logistics_provider", "carrier_name"),
            "carrier_id": pick(data, "carrier_id", "logistics_provider_id"),
            "status": str(pick(data, "shipment_status", "status") or "").lower() or None,
            "delivered_carrier_ts": _iso(
                pick(
                    data,
                    "delivered_carrier_date",
                    "order_delivered_carrier_date",
                    "handed_to_carrier_at",
                    "picked_up_at",
                )
            ),
            "delivered_customer_ts": _iso(
                pick(
                    data,
                    "delivered_customer_date",
                    "order_delivered_customer_date",
                    "delivered_at",
                )
            ),
            "estimated_ts": _iso(
                pick(data, "estimated_delivery_date", "order_estimated_delivery_date", "eta")
            ),
            "shipping_limit_ts": _iso(pick(data, "shipping_limit_date", "shipping_limit")),
            "delay_reason": pick(data, "delay_reason", "exception", "delay_cause"),
        }
        return finding
