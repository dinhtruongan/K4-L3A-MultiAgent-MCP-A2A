"""Offline tests for the multi-agent workflow using an in-memory fake gateway.

The fixtures are synthetic (same shape as the gateway tools, invented values) and never
leave the test process; the real evidence refs are issued by the MCP Gateway at run time.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.contracts import Contracts
from student_agent.mcp_gateway import MCPToolError, ToolSpec
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case

ROOT = Path(__file__).resolve().parents[1]
ORDER_ID = "0" * 31 + "1"
SELLER_ID = "seller-test-000001"
CASE_ID = "L3A_CASE_001"
OPENED_AT = "2018-01-15T09:00:00-03:00"
OUT_OF_SCOPE = "2018-06-01T10:00:00-03:00"  # after opened_at: must be ignored

POLICY = {
    "currency": "BRL",
    "policy_version": "TEST_POLICY",
    "rules": {
        "canceled_order_paid": {
            "case_status": "action_required",
            "recommended_action": "issue_refund",
            "refund_brl": 100.0,
            "responsible_parties": [{"party_id": None, "party_type": "platform"}],
        },
        "late_delivery_seller": {
            "case_status": "action_required",
            "recommended_action": "refund_freight",
            "refund_brl": 20.0,
            "responsible_parties": [{"party_id": "seller-from-template", "party_type": "seller"}],
        },
        "late_delivery_logistics": {
            "case_status": "action_required",
            "recommended_action": "refund_freight",
            "refund_brl": 20.0,
            "responsible_parties": [{"party_id": None, "party_type": "logistics_provider"}],
        },
        "duplicate_charge": {
            "case_status": "action_required",
            "recommended_action": "refund_duplicate_charge",
            "refund_brl": 70.0,
            "responsible_parties": [{"party_id": None, "party_type": "payment_provider"}],
        },
        "valid_split_payment": {
            "case_status": "no_action",
            "recommended_action": "document_no_action",
            "refund_brl": 0.0,
            "responsible_parties": [{"party_id": None, "party_type": "customer"}],
        },
        "refund_pending": {
            "case_status": "needs_investigation",
            "recommended_action": "monitor_refund",
            "refund_brl": 0.0,
            "responsible_parties": [{"party_id": None, "party_type": "payment_provider"}],
        },
        "unsupported_claim": {
            "case_status": "no_action",
            "recommended_action": "document_no_action",
            "refund_brl": 0.0,
            "responsible_parties": [{"party_id": None, "party_type": "customer"}],
        },
    },
}


def capture(amount: float, at: str = "2018-01-01T10:00:00-03:00") -> dict[str, Any]:
    return {"event_at": at, "event_type": "captured", "amount_brl": f"{amount:.2f}"}


BASE: dict[str, Any] = {
    "order": {
        "order_id": ORDER_ID,
        "customer_id": "customer-test",
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-01-01T09:00:00-03:00",
        "order_approved_at": "2018-01-01T10:00:00-03:00",
        "order_delivered_carrier_date": "2018-01-03T09:00:00-03:00",
        "order_delivered_customer_date": "2018-01-08T09:00:00-03:00",
        "order_estimated_delivery_date": "2018-01-10T09:00:00-03:00",
    },
    "item": [
        {
            "order_id": ORDER_ID,
            "order_item_id": "item-test",
            "product_id": "product-test",
            "seller_id": SELLER_ID,
            "shipping_limit_date": "2018-01-04T09:00:00-03:00",
            "price": "100.00",
            "freight_value": "20.00",
        },
        {  # a row from outside the case window (must not change any conclusion)
            "order_id": ORDER_ID,
            "order_item_id": "item-test",
            "product_id": "product-test",
            "seller_id": SELLER_ID,
            "shipping_limit_date": OUT_OF_SCOPE,
            "price": "100.00",
            "freight_value": "99.00",
        },
    ],
    "payment": {
        "order_id": ORDER_ID,
        "payments": [
            {"payment_sequential": "1", "payment_type": "credit_card", "payment_value": "120.00"},
            {"payment_sequential": "1", "payment_type": "credit_card", "payment_value": "33.00"},
        ],
        "events": [capture(120.0), capture(33.0, OUT_OF_SCOPE)],
    },
    "shipment": {
        "order_id": ORDER_ID,
        "order_status": "delivered",
        "delivered_carrier_at": "2018-01-03T09:00:00-03:00",
        "delivered_customer_at": "2018-01-08T09:00:00-03:00",
        "estimated_delivery_at": "2018-01-10T09:00:00-03:00",
        "shipping_limits": [
            {"seller_id": SELLER_ID, "shipping_limit_at": "2018-01-04T09:00:00-03:00"},
            {"seller_id": SELLER_ID, "shipping_limit_at": OUT_OF_SCOPE},
        ],
        "events": [],
    },
    "seller": [{"seller_id": SELLER_ID, "seller_state": "SP"}],
    "policy": POLICY,
}

ORDER_ARGS = ("case_id", "order_id")
TOOLS = {
    "get_order": ("order", ORDER_ARGS),
    "get_order_items": ("item", ORDER_ARGS),
    "get_payment_timeline": ("payment", ORDER_ARGS),
    "get_refund_timeline": ("refund", ORDER_ARGS),
    "get_shipment_summary": ("shipment", ORDER_ARGS),
    "get_sellers": ("seller", ORDER_ARGS),
    "get_policy": ("policy", ("case_id", "policy_version")),
    "get_customer_history": ("customer", ("case_id", "customer_unique_id")),
}


class FakeGateway:
    def __init__(self, data: dict[str, Any], failures: dict[str, list[Exception]] | None = None):
        self.data = data
        self.failures = failures or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def tool_specs(self) -> dict[str, ToolSpec]:
        return {
            name: ToolSpec(name=name, properties=props, required=props)
            for name, (_, props) in TOOLS.items()
        }

    async def list_tools(self) -> list[str]:
        return sorted(TOOLS)

    async def call(self, tool_name: str, *, case_id: str, **arguments: Any) -> dict[str, Any]:
        self.calls.append((tool_name, {"case_id": case_id, **arguments}))
        pending = self.failures.get(tool_name)
        if pending:
            raise pending.pop(0)
        domain = TOOLS[tool_name][0]
        if domain not in self.data:
            raise MCPToolError(tool_name, f"Error executing tool {tool_name}")
        digest = hashlib.sha256(f"{case_id}:{tool_name}:{arguments}".encode()).hexdigest()
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": f"ev_{digest[:32]}",
            "result_hash": f"sha256:{digest}",
            "domain": domain,
            "data": self.data[domain],
        }


def make_case(topic: str) -> dict[str, Any]:
    return {
        "case_id": CASE_ID,
        "opened_at": OPENED_AT,
        "customer_request": {
            "language": "vi",
            "message": "Không mặc định lời khai là đúng.",
            "claimed_order_id": ORDER_ID,
            "claims": [
                {"claim_id": "claim-001-a", "topic": topic},
                {"claim_id": "claim-001-b", "topic": "requested_full_refund"},
            ],
        },
        "policy_version": "TEST_POLICY",
    }


def run_case(tmp_path: Path, data: dict[str, Any], topic: str, **kwargs: Any):
    contracts = Contracts(ROOT / "contracts" / "schemas")
    trace_path = tmp_path / "trace.jsonl"
    trace = TraceWriter(trace_path, contracts)
    gateway = FakeGateway(data, kwargs.pop("failures", None))
    output = asyncio.run(solve_case(make_case(topic), gateway, trace))  # type: ignore[arg-type]
    contracts.validate_output(output, "output")
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    return output, events, gateway


def scenario(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(BASE)
    for dotted, value in changes.items():
        domain, _, field = dotted.partition("__")
        if field:
            data[domain][field] = value
        elif value is None:
            data.pop(domain)
        else:
            data[domain] = value
    return data


def verdicts(output: dict[str, Any]) -> dict[str, str]:
    return {c["claim_id"]: c["verdict"] for c in output["claim_assessments"]}


def test_canceled_paid_order_uses_policy_rule(tmp_path: Path) -> None:
    output, events, gateway = run_case(
        tmp_path, scenario(order__order_status="canceled"), "canceled_order_paid"
    )
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
    assert output["assessment"]["case_status"] == "action_required"
    assert output["financial_resolution"]["recommended_refund_brl"] == 100.0
    assert output["resolution_actions"] == ["issue_refund"]
    assert output["root_cause_analysis"]["responsible_parties"] == [
        {"party_type": "platform", "party_id": None}
    ]
    assert verdicts(output) == {"claim-001-a": "supported", "claim-001-b": "supported"}
    assert all(call[1]["case_id"] == CASE_ID for call in gateway.calls)
    assert gateway.calls[0] == ("get_order", {"case_id": CASE_ID, "order_id": ORDER_ID})
    assert ("get_policy", {"case_id": CASE_ID, "policy_version": "TEST_POLICY"}) in gateway.calls
    consumed = {
        ref
        for e in events
        if e["event_type"] == "tool_result_consumed"
        for ref in e["evidence_refs"]
    }
    assert set(output["evidence_refs"]) <= consumed


def test_lifecycle_events_are_ordered(tmp_path: Path) -> None:
    _, events, _ = run_case(tmp_path, scenario(), "unsupported_claim")
    kinds = [e["event_type"] for e in events]
    for required in (
        "task_assigned",
        "tool_result_consumed",
        "handoff",
        "policy_decided",
        "verification_completed",
    ):
        assert required in kinds
    assert kinds.index("policy_decided") < kinds.index("verification_completed")
    actors = {e["actor"] for e in events}
    assert {
        "coordinator",
        "order-agent",
        "payment-agent",
        "shipment-agent",
        "policy-agent",
        "verifier-agent",
    } <= actors


def late(handover: str) -> dict[str, Any]:
    data = scenario()
    for source, prefix in ((data["order"], "order_"), (data["shipment"], "")):
        key = "delivered_customer_date" if prefix else "delivered_customer_at"
        source[f"{prefix}{key}"] = "2018-01-13T09:00:00-03:00"
        key = "delivered_carrier_date" if prefix else "delivered_carrier_at"
        source[f"{prefix}{key}"] = handover
    return data


def test_late_delivery_seller_vs_logistics(tmp_path: Path) -> None:
    output, _, _ = run_case(tmp_path, late("2018-01-06T09:00:00-03:00"), "late_delivery_seller")
    assert output["assessment"]["primary_issue"] == "late_delivery_seller"
    # the policy names the role; the seller id comes from evidence, not the template
    assert output["root_cause_analysis"]["responsible_parties"] == [
        {"party_type": "seller", "party_id": SELLER_ID}
    ]
    assert output["financial_resolution"]["recommended_refund_brl"] == 20.0
    assert verdicts(output)["claim-001-b"] == "partially_supported"

    output, _, _ = run_case(
        tmp_path / "b", late("2018-01-03T09:00:00-03:00"), "late_delivery_seller"
    )
    assert output["assessment"]["primary_issue"] == "late_delivery_logistics"
    assert output["root_cause_analysis"]["responsible_parties"][0]["party_type"] == (
        "logistics_provider"
    )
    assert verdicts(output)["claim-001-a"] == "unsupported"
    assert output["assessment"]["confidence"] < 0.9


def test_duplicate_charge_and_split_payment(tmp_path: Path) -> None:
    dup = scenario(payment__events=[capture(70.0), capture(70.0), capture(50.0, OUT_OF_SCOPE)])
    output, _, _ = run_case(tmp_path, dup, "duplicate_charge")
    assert output["assessment"]["primary_issue"] == "duplicate_charge"
    assert output["financial_resolution"]["recommended_refund_brl"] == 70.0

    split = scenario(payment__events=[capture(70.0), capture(50.0), capture(60.0, OUT_OF_SCOPE)])
    output, _, _ = run_case(tmp_path / "b", split, "valid_split_payment")
    assert output["assessment"]["primary_issue"] == "valid_split_payment"
    assert output["assessment"]["case_status"] == "no_action"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0
    assert verdicts(output)["claim-001-b"] == "unsupported"


def test_out_of_scope_refund_is_ignored(tmp_path: Path) -> None:
    stale = {"event_at": OUT_OF_SCOPE, "event_type": "refund_requested", "status": "pending"}
    output, _, _ = run_case(tmp_path, scenario(refund={"events": [stale]}), "refund_pending")
    assert output["assessment"]["primary_issue"] == "unsupported_claim"

    fresh = {**stale, "event_at": "2018-01-14T09:00:00-03:00", "amount_brl": "120.00"}
    output, _, _ = run_case(tmp_path / "b", scenario(refund={"events": [fresh]}), "refund_pending")
    assert output["assessment"]["primary_issue"] == "refund_pending"
    assert output["assessment"]["case_status"] == "needs_investigation"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0


def test_unsupported_claim_when_nothing_is_wrong(tmp_path: Path) -> None:
    output, _, _ = run_case(tmp_path, scenario(), "unsupported_claim")
    assert output["assessment"]["primary_issue"] == "unsupported_claim"
    assert output["assessment"]["case_status"] == "no_action"
    assert output["financial_resolution"]["refund_lines"] == []
    assert verdicts(output) == {"claim-001-a": "unsupported", "claim-001-b": "unsupported"}


def test_missing_order_evidence_is_insufficient(tmp_path: Path) -> None:
    output, _, _ = run_case(tmp_path, scenario(order=None), "late_delivery_seller")
    assert output["assessment"]["primary_issue"] == "insufficient_evidence"
    assert output["assessment"]["case_status"] == "needs_investigation"
    assert output["assessment"]["confidence"] <= 0.4


def test_transient_errors_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("student_agent.specialists.BACKOFF_SECONDS", (0, 0))
    failures = {"get_order": [RuntimeError("timeout")]}
    output, _, gateway = run_case(
        tmp_path, scenario(order__order_status="canceled"), "canceled_order_paid", failures=failures
    )
    assert [c[0] for c in gateway.calls].count("get_order") == 2
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"


def test_tool_errors_are_not_retried(tmp_path: Path) -> None:
    _, _, gateway = run_case(tmp_path, scenario(), "unsupported_claim")
    assert [c[0] for c in gateway.calls].count("get_refund_timeline") == 1
