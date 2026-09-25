"""Offline tests for the multi-agent workflow using an in-memory fake gateway.

The fixtures are synthetic and never leave the test process; the real evidence refs are
issued by the MCP Gateway at run time.
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
SELLER_ID = "a" * 32

BASE: dict[str, dict[str, Any]] = {
    "order": {
        "order_id": ORDER_ID,
        "customer_id": "c" * 32,
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-01-01 10:00:00",
        "order_approved_at": "2018-01-01 11:00:00",
        "order_delivered_carrier_date": "2018-01-03 10:00:00",
        "order_delivered_customer_date": "2018-01-08 10:00:00",
        "order_estimated_delivery_date": "2018-01-20 00:00:00",
        "items": [
            {
                "order_item_id": 1,
                "product_id": "p" * 32,
                "seller_id": SELLER_ID,
                "shipping_limit_date": "2018-01-05 10:00:00",
                "price": 100.0,
                "freight_value": 20.0,
            }
        ],
    },
    "payment": {
        "order_id": ORDER_ID,
        "payments": [
            {"payment_sequential": 1, "payment_type": "credit_card", "payment_value": 120.0}
        ],
    },
    "shipment": {"order_id": ORDER_ID, "shipment_id": "SHP-1", "carrier": "correios"},
    "seller": {"seller_id": SELLER_ID, "seller_state": "SP"},
    "policy": {"policy_id": "POL-1", "summary": "generic dispute policy"},
}

TOOLS = {
    "get_order": ("order", ("case_id", "order_id")),
    "get_payment": ("payment", ("case_id", "order_id")),
    "get_shipment": ("shipment", ("case_id", "order_id")),
    "get_seller": ("seller", ("case_id", "seller_id")),
    "get_policy": ("policy", ("case_id",)),
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
            raise MCPToolError(tool_name, "not found")
        digest = hashlib.sha256(f"{case_id}:{tool_name}:{arguments}".encode()).hexdigest()
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": f"ev_{digest[:32]}",
            "result_hash": f"sha256:{digest}",
            "domain": domain,
            "data": self.data[domain],
        }


def run_case(tmp_path: Path, data: dict[str, Any], message: str, **kwargs: Any):
    contracts = Contracts(ROOT / "contracts" / "schemas")
    trace_path = tmp_path / "trace.jsonl"
    trace = TraceWriter(trace_path, contracts)
    gateway = FakeGateway(data, kwargs.pop("failures", None))
    case = {"case_id": "L3A_CASE_001", "order_id": ORDER_ID, "customer_message": message}
    output = asyncio.run(solve_case(case, gateway, trace))  # type: ignore[arg-type]
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


def test_canceled_paid_order_refunds_everything(tmp_path: Path) -> None:
    output, events, gateway = run_case(
        tmp_path, scenario(order__order_status="canceled"), "Pedido cancelado e fui cobrado"
    )
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
    assert output["assessment"]["case_status"] == "action_required"
    assert output["financial_resolution"]["recommended_refund_brl"] == 120.0
    assert all(call[1]["case_id"] == "L3A_CASE_001" for call in gateway.calls)
    consumed = {
        ref
        for e in events
        if e["event_type"] == "tool_result_consumed"
        for ref in e["evidence_refs"]
    }
    assert set(output["evidence_refs"]) <= consumed


def test_lifecycle_events_are_ordered(tmp_path: Path) -> None:
    _, events, _ = run_case(tmp_path, scenario(), "meu pedido atrasou")
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


def test_late_delivery_seller_vs_logistics(tmp_path: Path) -> None:
    late = {"order_delivered_customer_date": "2018-01-25 10:00:00"}
    seller_late = scenario(
        order={**BASE["order"], **late, "order_delivered_carrier_date": "2018-01-10 10:00:00"}
    )
    output, _, _ = run_case(tmp_path, seller_late, "the order arrived late")
    assert output["assessment"]["primary_issue"] == "late_delivery_seller"
    assert output["root_cause_analysis"]["responsible_parties"] == [
        {"party_type": "seller", "party_id": SELLER_ID}
    ]
    carrier_late = scenario(order={**BASE["order"], **late})
    output, _, _ = run_case(tmp_path / "b", carrier_late, "the order arrived late")
    assert output["assessment"]["primary_issue"] == "late_delivery_logistics"
    assert output["root_cause_analysis"]["responsible_parties"][0]["party_type"] == (
        "logistics_provider"
    )


def test_duplicate_charge_and_split_payment(tmp_path: Path) -> None:
    dup = scenario(
        payment={
            "payments": [
                {"payment_sequential": 1, "payment_type": "credit_card", "payment_value": 120.0},
                {"payment_sequential": 2, "payment_type": "credit_card", "payment_value": 120.0},
            ]
        }
    )
    output, _, _ = run_case(tmp_path, dup, "fui cobrado duas vezes")
    assert output["assessment"]["primary_issue"] == "duplicate_charge"
    assert output["financial_resolution"]["recommended_refund_brl"] == 120.0

    split = scenario(
        payment={
            "payments": [
                {"payment_sequential": 1, "payment_type": "voucher", "payment_value": 20.0},
                {"payment_sequential": 2, "payment_type": "credit_card", "payment_value": 100.0},
            ]
        }
    )
    output, _, _ = run_case(tmp_path / "b", split, "fui cobrado duas vezes")
    assert output["assessment"]["primary_issue"] == "valid_split_payment"
    assert output["assessment"]["case_status"] == "no_action"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0


def test_unsupported_claim_when_delivery_on_time(tmp_path: Path) -> None:
    output, _, _ = run_case(tmp_path, scenario(), "my order is late")
    assert output["assessment"]["primary_issue"] == "unsupported_claim"
    assert output["financial_resolution"]["refund_lines"] == []


def test_missing_order_evidence_is_insufficient(tmp_path: Path) -> None:
    output, _, _ = run_case(tmp_path, scenario(order=None), "my order is late")
    assert output["assessment"]["primary_issue"] == "insufficient_evidence"
    assert output["assessment"]["case_status"] == "needs_investigation"
    assert output["assessment"]["confidence"] <= 0.4


def test_transient_errors_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("student_agent.specialists.BACKOFF_SECONDS", (0, 0))
    failures = {"get_order": [RuntimeError("timeout")]}
    output, _, gateway = run_case(
        tmp_path, scenario(order__order_status="canceled"), "cancelado", failures=failures
    )
    assert [c[0] for c in gateway.calls].count("get_order") == 2
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
