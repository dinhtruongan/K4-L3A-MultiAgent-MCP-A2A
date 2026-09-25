"""Specialist agents. Each one owns a fixed set of MCP tool domains (least privilege).

Every specialist applies the same *temporal scope*: a row is only used when its own
timestamp falls inside the order's lifetime known at claim time, i.e. between the order
purchase and the case ``opened_at``. Rows outside that window are kept out of the facts
(and counted in ``excluded``) because the gateway may return rows that do not belong to
the scenario under investigation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime, timedelta
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
SCOPE_TOLERANCE = timedelta(hours=1)

# Accepted names for each argument a tool may declare. Values come only from the case
# input or from earlier evidence; nothing is invented.
ARGUMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "order_id": ("order_id",),
    "customer_id": ("customer_id",),
    "customer_unique_id": ("customer_unique_id",),
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
    # A tool error is an answer from the gateway (deterministic); only transport-level
    # failures are worth retrying.
    return not isinstance(exc, MCPToolError | ContractError | ValueError | ToolUnavailable)


class Scope:
    """Temporal window ``[purchase - tolerance, opened_at]`` for one case."""

    def __init__(self, start: datetime | None, end: datetime | None) -> None:
        self.start = start - SCOPE_TOLERANCE if start else None
        self.end = end

    def contains(self, value: Any) -> bool:
        moment = to_datetime(value) if not isinstance(value, datetime) else value
        if moment is None:
            return True  # undated rows cannot be excluded on time grounds
        if self.start and moment < self.start:
            return False
        return not (self.end and moment > self.end)

    def split(self, rows: Iterable[dict[str, Any]], *fields: str) -> tuple[list, list]:
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for row in rows:
            (kept if self.contains(pick_shallow(row, *fields)) else dropped).append(row)
        return kept, dropped


def case_scope(state: CaseState) -> Scope:
    order = state.facts("order-agent")
    return Scope(to_datetime(order.get("purchase_ts")), to_datetime(state.case.get("opened_at")))


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
            elif isinstance(error, MCPToolError):
                code = "no_records"
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
    item_id = pick_shallow(raw, "order_item_id", "item_id", "order_item_ref", "id")
    return {
        "item_id": str(item_id) if item_id is not None else None,
        "product_id": pick_shallow(raw, "product_id"),
        "seller_id": pick_shallow(raw, "seller_id"),
        "price": price,
        "freight": freight,
        "shipping_limit": _iso(pick_shallow(raw, "shipping_limit_date", "shipping_limit")),
    }


class OrderItemAgent(Specialist):
    name = "order-agent"
    tools = {
        "order": ("get_order", "get_order_details"),
        "item": ("get_order_items", "get_items", "list_order_items"),
        "seller": ("get_sellers", "get_seller", "get_seller_profile"),
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
        }
        scope = Scope(to_datetime(facts["purchase_ts"]), to_datetime(state.case.get("opened_at")))
        context = {**hints, "order_id": order_id, "customer_id": facts["customer_id"]}

        raw_items = records(pick_shallow(data, "items", "order_items") or [], marker="price")
        if not raw_items:
            item_evidence = await self.fetch(state, finding, "item", context, "item_prices")
            if item_evidence:
                raw_items = records(item_evidence.data, "items", "order_items", marker="price")
        kept, dropped = scope.split(raw_items, "shipping_limit_date", "shipping_limit")
        if raw_items and not kept:
            kept, dropped = raw_items, []  # never discard the whole order
        items = [_item_record(raw) for raw in kept]
        facts["items"] = items
        facts["items_excluded"] = len(dropped)
        facts["item_totals"] = [
            round((i["price"] or 0) + (i["freight"] or 0), 2) for i in items if i["price"]
        ]
        facts["order_total"] = round(sum(facts["item_totals"]), 2) if items else None
        facts["freight_total"] = round(sum(i["freight"] or 0 for i in items), 2)
        facts["seller_ids"] = strings(
            [*(i["seller_id"] for i in items), pick(data, "seller_id"), hints.get("seller_id")]
        )
        limits = [to_datetime(i["shipping_limit"]) for i in items if i["shipping_limit"]]
        facts["shipping_limit_ts"] = max(limits).isoformat() if limits else None

        sellers: dict[str, dict[str, Any]] = {}
        if message.payload.get("need_seller", True) and order_id:
            evidence = await self.fetch(state, finding, "seller", context, "seller_profile")
            for raw in records(evidence.data if evidence else [], "sellers", marker="seller_id"):
                seller_id = pick_shallow(raw, "seller_id")
                if seller_id in facts["seller_ids"]:
                    sellers[seller_id] = {"state": pick_shallow(raw, "seller_state", "state")}
        facts["sellers"] = sellers
        finding.facts = facts
        return finding


def _events(data: Any) -> list[dict[str, Any]]:
    return records(pick_shallow(data, "events", "lifecycle", "timeline") or [], marker="event_at")


def _amount(raw: dict[str, Any]) -> float | None:
    return to_float(pick_shallow(raw, "amount_brl", "payment_value", "amount", "value"))


class PaymentAgent(Specialist):
    name = "payment-agent"
    tools = {
        "payment": ("get_payment_timeline", "get_order_payments", "get_payments"),
        "refund": ("get_refund_timeline", "get_refunds", "get_refund_status"),
    }

    async def run(self, state: CaseState, message: AgentMessage) -> Finding:
        finding = Finding(agent=self.name)
        scope = case_scope(state)
        context = {**state.hints, **message.payload.get("context", {})}
        payment = await self.fetch(state, finding, "payment", context, "payment_ledger")
        data = payment.data if payment else []
        base_rows = records(pick_shallow(data, "payments") or data, marker="payment_value")
        base_rows = [row for row in base_rows if pick_shallow(row, "payment_value") is not None]
        events = _events(data)
        kept_events, dropped_events = scope.split(events, "event_at", "timestamp", "created_at")

        captures: list[dict[str, Any]] = []
        lifecycle: list[dict[str, Any]] = []
        unmatched = list(base_rows)
        for raw in kept_events:
            kind = str(pick_shallow(raw, "event_type", "type") or "").lower()
            status = str(pick_shallow(raw, "status") or "").lower() or None
            amount = _amount(raw)
            if kind in {"captured", "capture", "payment_captured", "paid"}:
                if status in {"failed", "declined", "voided", "reversed"} or amount is None:
                    continue
                base = next((r for r in unmatched if _amount(r) == amount), None)
                if base is not None:
                    unmatched.remove(base)
                captures.append(
                    {
                        "value": amount,
                        "at": _iso(pick_shallow(raw, "event_at")),
                        "type": pick_shallow(base or {}, "payment_type", "type", "method"),
                        "sequential": pick_shallow(base or {}, "payment_sequential"),
                    }
                )
            else:
                lifecycle.append({"type": kind, "status": status, "amount": amount})
        if not events:  # plain payment rows without a lifecycle: nothing to scope on
            for raw in base_rows:
                captures.append(
                    {
                        "value": _amount(raw),
                        "at": None,
                        "type": pick_shallow(raw, "payment_type", "type", "method"),
                        "sequential": pick_shallow(raw, "payment_sequential"),
                    }
                )

        refunds: list[dict[str, Any]] = []
        refunds_excluded = 0
        if message.payload.get("need_refund") and self.tool_for("refund") is not None:
            refund = await self.fetch(state, finding, "refund", context, "refund_status")
            if refund:
                raw_refunds = _events(refund.data) or records(refund.data, "refunds")
                kept, dropped = scope.split(raw_refunds, "event_at", "requested_at", "created_at")
                refunds = list(_refunds_from(kept))
                refunds_excluded = len(dropped)
        finding.facts = {
            "payment_found": payment is not None,
            "captures": captures,
            "lifecycle": lifecycle,
            "paid_total": round(sum(c["value"] for c in captures), 2) if payment else None,
            "payment_count": len(captures),
            "payment_types": unique(c["type"] for c in captures),
            "payment_references": [],
            "payments_excluded": len(dropped_events),
            "ledger_rows": len(base_rows),
            "refunds": refunds,
            "refunds_excluded": refunds_excluded,
        }
        return finding


def _refunds_from(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    result = []
    for raw in rows:
        status = pick_shallow(raw, "refund_status", "status", "state")
        amount = to_float(pick_shallow(raw, "amount_brl", "amount", "refund_amount", "value"))
        if status is None and amount is None:
            continue
        result.append(
            {
                "refund_id": pick_shallow(raw, "refund_id", "id", "reference"),
                "status": str(status or "").lower() or None,
                "amount": amount,
                "event": pick_shallow(raw, "event_type"),
                "at": _iso(pick_shallow(raw, "event_at", "requested_at", "created_at")),
            }
        )
    return result


class ShipmentAgent(Specialist):
    name = "shipment-agent"
    tools = {"shipment": ("get_shipment_summary", "get_shipment", "get_tracking", "get_delivery")}

    async def run(self, state: CaseState, message: AgentMessage) -> Finding:
        finding = Finding(agent=self.name)
        scope = case_scope(state)
        context = {**state.hints, **message.payload.get("context", {})}
        shipment = await self.fetch(state, finding, "shipment", context, "delivery_timeline")
        data = shipment.data if shipment else {}
        limits_raw = records(pick_shallow(data, "shipping_limits") or [], marker="seller_id")
        kept_limits, dropped_limits = scope.split(
            limits_raw, "shipping_limit_at", "shipping_limit_date"
        )
        kept_events, dropped_events = scope.split(_events(data), "event_at")
        limits = [
            to_datetime(pick_shallow(row, "shipping_limit_at", "shipping_limit_date"))
            for row in kept_limits
        ]
        limits = [value for value in limits if value]
        finding.facts = {
            "shipment_found": shipment is not None,
            "shipment_ids": strings([pick_shallow(data, "shipment_id", "tracking_id")]),
            "carrier": pick_shallow(data, "carrier", "logistics_provider", "carrier_name"),
            "status": str(pick_shallow(data, "order_status", "status") or "").lower() or None,
            "delivered_carrier_ts": _iso(
                pick_shallow(data, "delivered_carrier_at", "delivered_carrier_date")
            ),
            "delivered_customer_ts": _iso(
                pick_shallow(data, "delivered_customer_at", "delivered_customer_date")
            ),
            "estimated_ts": _iso(
                pick_shallow(data, "estimated_delivery_at", "estimated_delivery_date")
            ),
            "shipping_limit_ts": max(limits).isoformat() if limits else None,
            "events": [
                {
                    "type": str(pick_shallow(row, "event_type") or "").lower(),
                    "actor": pick_shallow(row, "actor"),
                    "status": pick_shallow(row, "status"),
                }
                for row in kept_events
            ],
            "excluded": len(dropped_limits) + len(dropped_events),
        }
        return finding
