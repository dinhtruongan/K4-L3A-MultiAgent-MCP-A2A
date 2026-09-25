"""Verifier agent: cross-field invariants and confidence calibration."""

from __future__ import annotations

from dataclasses import replace

from .evidence import money
from .policy import CAUSE_CODES, DEFAULT_RULES, ISSUE_DOMAINS
from .state import CaseState, Decision

BASE_CONFIDENCE = 0.93
INSUFFICIENT_CONFIDENCE = 0.3

# Evidence domains without which the conclusion is not allowed to stand.
REQUIRED_DOMAINS: dict[str, tuple[tuple[str, ...], ...]] = {
    "canceled_order_paid": (("order",), ("payment",)),
    "unavailable_order_paid": (("order",), ("payment",)),
    "late_delivery_seller": (("order",), ("shipment", "item")),
    "late_delivery_logistics": (("order",), ("shipment", "item")),
    "valid_split_payment": (("order",), ("payment",)),
    "payment_mismatch": (("order",), ("payment",)),
    "duplicate_charge": (("order",), ("payment",)),
    "refund_pending": (("order",), ("refund",)),
    "refund_failed": (("order",), ("refund",)),
    "unsupported_claim": (("order",),),
    "insufficient_evidence": (),
}

NOTE_PENALTY = {
    "handover_timestamps_missing": 0.15,
    "late_actor_disagrees": 0.15,
    "policy_rule_missing": 0.1,
}

CLAIM_ISSUES = {
    "cancel": {"canceled_order_paid", "refund_pending", "refund_failed"},
    "unavailable": {"unavailable_order_paid", "refund_pending", "refund_failed"},
    "late": {"late_delivery_seller", "late_delivery_logistics"},
    "duplicate": {"duplicate_charge", "valid_split_payment", "payment_mismatch"},
    "mismatch": {"payment_mismatch", "valid_split_payment", "duplicate_charge"},
    "refund": {"refund_pending", "refund_failed", "canceled_order_paid", "unavailable_order_paid"},
}


def _downgrade(decision: Decision, reason: str) -> Decision:
    issue = "insufficient_evidence"
    return replace(
        decision,
        primary_issue=issue,
        case_status=DEFAULT_RULES[issue]["case_status"],
        ranked_causes=list(CAUSE_CODES[issue]),
        responsible_parties=[("unknown", None)],
        refund_lines=[],
        resolution_actions=[DEFAULT_RULES[issue]["action"]],
        evidence_domains=list(ISSUE_DOMAINS[issue]),
        rule=f"{decision.rule}->VERIFIER_{reason}",
        confidence_notes=[*decision.confidence_notes, reason.lower()],
    )


def verify(state: CaseState, decision: Decision) -> tuple[Decision, list[str]]:
    """Return the (possibly corrected) decision and the list of check codes applied."""
    checks: list[str] = []
    available = {item.domain for item in state.ledger.values()}

    for alternatives in REQUIRED_DOMAINS.get(decision.primary_issue, ()):
        if not available & set(alternatives):
            reason = f"MISSING_{alternatives[0].upper()}_EVIDENCE"
            checks.append(reason)
            decision = _downgrade(decision, reason)
            break

    order = state.facts("order-agent")
    payment = state.facts("payment-agent")

    # Money: non-negative lines, rounded, never above what was actually paid (in scope).
    paid = payment.get("paid_total")
    lines = [(code, money(amount), entity) for code, amount, entity in decision.refund_lines]
    total = sum(amount for _, amount, _ in lines)
    if paid is not None and total > paid + 0.01:
        checks.append("REFUND_CAPPED_AT_PAID_TOTAL")
        scale = paid / total if total else 0.0
        lines = [(code, money(amount * scale), entity) for code, amount, entity in lines]
    decision.refund_lines = [line for line in lines if line[1] > 0]

    # Status / refund / action consistency.
    if decision.case_status == "no_action" and decision.refund_lines:
        checks.append("NO_ACTION_WITH_REFUND")
        decision.refund_lines = []
    if decision.refund_lines and decision.case_status != "action_required":
        checks.append("REFUND_REQUIRES_ACTION")
        decision.case_status = "action_required"
    if not decision.resolution_actions:
        checks.append("ADDED_DEFAULT_ACTION")
        decision.resolution_actions = [DEFAULT_RULES[decision.primary_issue]["action"]]
    decision.resolution_actions = list(dict.fromkeys(decision.resolution_actions))[:8]

    # Responsibility: seller IDs must come from evidence; seller fault excludes carrier.
    known_sellers = set(order.get("seller_ids") or [])
    parties = []
    for party_type, party_id in decision.responsible_parties:
        if party_type == "seller" and party_id is not None and party_id not in known_sellers:
            checks.append("DROPPED_UNKNOWN_SELLER")
            continue
        parties.append((party_type, party_id))
    types = {party_type for party_type, _ in parties}
    if "seller" in types and "logistics_provider" in types:
        checks.append("SELLER_FAULT_EXCLUDES_CARRIER")
        parties = [party for party in parties if party[0] != "logistics_provider"]
    decision.responsible_parties = list(dict.fromkeys(parties))[:5] or [("unknown", None)]

    decision.confidence = calibrate(state, decision)
    checks.append("CONFIDENCE_CALIBRATED")
    return decision, checks


def calibrate(state: CaseState, decision: Decision) -> float:
    if decision.primary_issue == "insufficient_evidence":
        return INSUFFICIENT_CONFIDENCE
    confidence = BASE_CONFIDENCE
    cited = [item for item in state.ledger.values() if item.domain in decision.evidence_domains]
    confidence -= min(0.15, 0.05 * sum(len(item.warnings) for item in cited))
    confidence -= 0.05 * len(decision.data_conflicts)
    for note in decision.confidence_notes:
        confidence -= NOTE_PENALTY.get(note, 0.05)
    hard_errors = [
        error
        for finding in state.findings.values()
        for error in finding.errors
        if error.split(":", 1)[0] in {"tool_error", "invalid_envelope", "forbidden"}
    ]
    confidence -= min(0.1, 0.05 * len(hard_errors))
    claimed = state.hints.get("claimed_issues") or []
    claims = state.hints.get("claims") or []
    if claimed:
        if decision.primary_issue not in claimed:
            confidence -= 0.2
    elif claims:
        related = set().union(*(CLAIM_ISSUES.get(claim, set()) for claim in claims))
        if decision.primary_issue not in related | {"unsupported_claim"}:
            confidence -= 0.1
    return round(min(0.95, max(0.05, confidence)), 2)
