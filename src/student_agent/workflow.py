"""Coordinator for the L3A multi-agent workflow.

The coordinator never touches a tool or decides an outcome. It assigns each
question to the specialist that owns it, hands the collected facts to the policy
agent, and lets the verifier veto the result before the case is closed.
"""

from __future__ import annotations

from typing import Any

from .agents import (
    COORDINATOR,
    ORDER_AGENT,
    PAYMENT_AGENT,
    POLICY_AGENT,
    SHIPMENT_AGENT,
    VERIFIER,
    EvidenceCollector,
    EvidenceCollectorError,
    decide,
    investigate_order,
    investigate_payments,
    investigate_shipment,
    load_policy,
    safe_fallback,
    verify,
)
from .llm import classify
from .mcp_gateway import EvidenceGateway
from .policy import scope_evidence
from .trace import TraceWriter


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Run one case through the coordinator, specialists, policy agent and verifier."""
    case_id = str(case["case_id"])
    request = case.get("customer_request") or {}
    order_id = str(request.get("claimed_order_id") or "")
    policy_version = str(case.get("policy_version") or "")
    collector = EvidenceCollector(gateway=gateway, trace=trace, case_id=case_id)

    def assign(target: str, decision_code: str) -> None:
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor=COORDINATOR,
            target=target,
            decision_code=decision_code,
        )

    def handoff(actor: str, target: str, decision_code: str) -> None:
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor=actor,
            target=target,
            decision_code=decision_code,
        )

    if not order_id:
        raise EvidenceCollectorError(f"{case_id}: customer request carries no claimed_order_id")

    assign(ORDER_AGENT, "VERIFY_ORDER_STATUS")
    order, items = await investigate_order(collector, order_id)
    handoff(ORDER_AGENT, COORDINATOR, "ORDER_FACTS_READY")

    assign(PAYMENT_AGENT, "VERIFY_PAYMENT_AND_REFUND")
    payment_timeline, refund_timeline = await investigate_payments(collector, order_id)
    handoff(PAYMENT_AGENT, COORDINATOR, "PAYMENT_FACTS_READY")

    assign(SHIPMENT_AGENT, "VERIFY_DELIVERY_COMMITMENT")
    shipment = await investigate_shipment(collector, order_id)
    handoff(SHIPMENT_AGENT, COORDINATOR, "SHIPMENT_FACTS_READY")

    scoped = scope_evidence(
        order=order,
        items=items,
        payment_timeline=payment_timeline,
        refund_timeline=refund_timeline,
        shipment=shipment,
        missing_tools=collector.missing_required,
    )

    handoff(COORDINATOR, POLICY_AGENT, "FACTS_COMPLETE")
    rules = await load_policy(collector, policy_version)
    claimed = {
        str(claim.get("topic", "")) for claim in request.get("claims") or []
    }
    classification = classify(scoped, claimed)
    output, decision_code = decide(case, scoped, rules, collector, classification)
    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor=POLICY_AGENT,
        decision_code=decision_code,
        attributes={
            "primary_issue": output["assessment"]["primary_issue"],
            "case_status": output["assessment"]["case_status"],
            "refund_brl": output["financial_resolution"]["recommended_refund_brl"],
            "dropped_rows": sum(scoped.dropped.values()),
            "reasoning_source": classification.source,
        },
    )
    handoff(POLICY_AGENT, VERIFIER, "DRAFT_READY")

    failures = verify(output, scoped)
    if failures:
        output = safe_fallback(output, scoped)
        failures = verify(output, scoped)
    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor=VERIFIER,
        decision_code="VERIFY_PASS" if not failures else f"VERIFY_FAIL_{failures[0]}",
        attributes={
            "failed_invariants": len(failures),
            # The auditor pass re-derives the label without seeing the analyst's
            # reasoning, so agreement here is a genuine second opinion.
            "second_opinion_agreed": classification.agreed,
        },
    )
    handoff(VERIFIER, COORDINATOR, "VERDICT_ACCEPTED" if not failures else "VERDICT_DOWNGRADED")
    return output
