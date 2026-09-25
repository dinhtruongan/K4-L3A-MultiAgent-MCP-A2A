"""Deterministic policy engine used by the policy agent.

``classify`` turns in-scope specialist facts into an issue; ``resolve`` applies the
machine-readable policy returned by ``get_policy`` (``rules[<issue>]``: case status,
recommended action, refund amount, responsible party). The customer message and claim
topics only steer confidence; they are never treated as ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from .evidence import close, money, pick, to_datetime, to_float, unique
from .state import CaseState, Decision

CANCELED = {"canceled", "cancelled"}
UNAVAILABLE = {"unavailable", "out_of_stock"}
REFUND_PENDING = {"pending", "processing", "requested", "in_progress", "created", "open"}
REFUND_FAILED = {"failed", "rejected", "declined", "error", "reversed"}
REFUND_DONE = {"completed", "succeeded", "success", "refunded", "processed", "paid", "done"}
OPEN_STATUSES = {"open", "pending", "unresolved", "confirmed", None}

CLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
    "cancel": (r"cancel", r"h[uủ]y", r"desist"),
    "unavailable": (
        r"indispon",
        r"unavailable",
        r"out of stock",
        r"sem estoque",
        r"h[ếe]t h[àa]ng",
    ),
    "late": (
        r"\blate\b",
        r"atras",
        r"delay",
        r"n[ãa]o (chegou|recebi)",
        r"not (arrived|received)",
        r"tr[ễe]",
        r"ch[ậa]m",
        r"prazo",
    ),
    "duplicate": (r"twice", r"duplicat", r"double", r"duas vezes", r"hai l[ầa]n"),
    "mismatch": (r"wrong amount", r"valor errado", r"overcharg", r"sai s[ốo] ti[ềe]n", r"diverg"),
    "refund": (r"refund", r"reembols", r"estorn", r"ho[àa]n ti[ềe]n"),
}

# Evidence domains cited for each conclusion: the groups that prove it, nothing else.
ISSUE_DOMAINS: dict[str, tuple[str, ...]] = {
    "canceled_order_paid": ("order", "payment", "policy"),
    "unavailable_order_paid": ("order", "item", "payment", "seller", "policy"),
    "late_delivery_seller": ("order", "item", "shipment", "seller", "policy"),
    "late_delivery_logistics": ("order", "shipment", "policy"),
    "valid_split_payment": ("order", "item", "payment", "policy"),
    "payment_mismatch": ("order", "item", "payment", "policy"),
    "duplicate_charge": ("order", "item", "payment", "policy"),
    "refund_pending": ("order", "payment", "refund", "policy"),
    "refund_failed": ("order", "payment", "refund", "policy"),
    "unsupported_claim": ("order", "payment", "shipment", "policy"),
    "insufficient_evidence": ("order", "item", "payment", "shipment", "refund"),
}

CAUSE_CODES: dict[str, list[str]] = {
    "canceled_order_paid": ["CANCELED_ORDER_PAID", "REFUND_NOT_ISSUED"],
    "unavailable_order_paid": ["UNAVAILABLE_ORDER_PAID", "SELLER_INVENTORY_UNAVAILABLE"],
    "late_delivery_seller": ["LATE_DELIVERY_SELLER", "SELLER_MISSED_SHIPPING_LIMIT"],
    "late_delivery_logistics": ["LATE_DELIVERY_LOGISTICS", "CARRIER_TRANSIT_DELAY"],
    "valid_split_payment": ["VALID_SPLIT_PAYMENT", "SPLIT_PAYMENT_MATCHES_ORDER_TOTAL"],
    "payment_mismatch": ["PAYMENT_MISMATCH", "RECONCILIATION_MISMATCH_OPEN"],
    "duplicate_charge": ["DUPLICATE_CHARGE", "DUPLICATE_PAYMENT_CAPTURE"],
    "refund_pending": ["REFUND_PENDING", "REFUND_AWAITING_PROCESSING"],
    "refund_failed": ["REFUND_FAILED", "REFUND_PROCESSING_FAILED"],
    "unsupported_claim": ["UNSUPPORTED_CLAIM", "CLAIM_CONTRADICTED_BY_EVIDENCE"],
    "insufficient_evidence": ["INSUFFICIENT_EVIDENCE", "MISSING_AUTHORITATIVE_EVIDENCE"],
}

# Used only when the policy evidence is unavailable or lacks a rule for the issue.
DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "canceled_order_paid": {
        "case_status": "action_required",
        "action": "issue_refund",
        "party": "platform",
    },
    "unavailable_order_paid": {
        "case_status": "action_required",
        "action": "issue_refund",
        "party": "seller",
    },
    "late_delivery_seller": {
        "case_status": "action_required",
        "action": "refund_freight",
        "party": "seller",
    },
    "late_delivery_logistics": {
        "case_status": "action_required",
        "action": "refund_freight",
        "party": "logistics_provider",
    },
    "valid_split_payment": {
        "case_status": "no_action",
        "action": "document_no_action",
        "party": "customer",
    },
    "payment_mismatch": {
        "case_status": "action_required",
        "action": "reconcile_payment",
        "party": "payment_provider",
    },
    "duplicate_charge": {
        "case_status": "action_required",
        "action": "refund_duplicate_charge",
        "party": "payment_provider",
    },
    "refund_pending": {
        "case_status": "needs_investigation",
        "action": "monitor_refund",
        "party": "payment_provider",
    },
    "refund_failed": {
        "case_status": "action_required",
        "action": "retry_refund",
        "party": "payment_provider",
    },
    "unsupported_claim": {
        "case_status": "no_action",
        "action": "document_no_action",
        "party": "customer",
    },
    "insufficient_evidence": {
        "case_status": "needs_investigation",
        "action": "request_additional_evidence",
        "party": "unknown",
    },
}

REFUND_REASON: dict[str, str] = {
    "canceled_order_paid": "CANCELED_ORDER_REFUND",
    "unavailable_order_paid": "UNAVAILABLE_ITEM_REFUND",
    "late_delivery_seller": "LATE_DELIVERY_FREIGHT_REFUND",
    "late_delivery_logistics": "LATE_DELIVERY_FREIGHT_REFUND",
    "payment_mismatch": "PAYMENT_MISMATCH_REFUND",
    "duplicate_charge": "DUPLICATE_CHARGE_REFUND",
    "refund_pending": "PENDING_REFUND",
    "refund_failed": "FAILED_REFUND_REISSUE",
}

PARTY_TYPES = {
    "seller",
    "platform",
    "logistics_provider",
    "payment_provider",
    "customer",
    "unknown",
}


@dataclass
class Classification:
    issue: str
    rule: str
    claims: list[str]
    details: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


TOPIC_CLAIMS: dict[str, tuple[str, ...]] = {
    "canceled_order_paid": ("cancel",),
    "unavailable_order_paid": ("unavailable",),
    "late_delivery_seller": ("late",),
    "late_delivery_logistics": ("late",),
    "valid_split_payment": ("duplicate",),
    "duplicate_charge": ("duplicate",),
    "payment_mismatch": ("mismatch",),
    "refund_pending": ("refund",),
    "refund_failed": ("refund",),
    "requested_full_refund": ("refund",),
    "unsupported_claim": (),
}


def claim_topics(case: dict[str, Any]) -> list[tuple[str, str]]:
    """``(claim_id, topic)`` pairs declared in the case input."""
    request = case.get("customer_request")
    source = request if isinstance(request, dict) else case
    claims = source.get("claims")
    if not isinstance(claims, list):
        return []
    return [
        (claim["claim_id"], str(claim.get("topic") or claim.get("type") or ""))
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("claim_id"), str)
    ]


def detect_claims(case: dict[str, Any], text: str) -> list[str]:
    """Claim categories. Structured claim topics win over keyword matching."""
    topics = [topic for _, topic in claim_topics(case) if topic]
    found = [category for topic in topics for category in TOPIC_CLAIMS.get(topic, ())]
    unknown = [topic for topic in topics if topic not in TOPIC_CLAIMS]
    if topics and not unknown:
        return unique(found)
    lowered = ("\n".join(unknown) if topics else text).lower()
    found.extend(
        claim
        for claim, patterns in CLAIM_PATTERNS.items()
        if any(re.search(pattern, lowered) for pattern in patterns)
    )
    return unique(found)


def _date(value: str | None):
    parsed = to_datetime(value)
    return parsed.date() if parsed else None


def _timeline(state: CaseState, cls: Classification) -> dict[str, Any]:
    """Order row vs shipment summary; shipment tracking wins on a >24 h disagreement."""
    order = state.facts("order-agent")
    shipment = state.facts("shipment-agent")
    merged: dict[str, Any] = {}
    for key, field_name in (
        ("delivered_customer_ts", "delivered_customer_date"),
        ("delivered_carrier_ts", "delivered_carrier_date"),
        ("estimated_ts", "estimated_delivery_date"),
        ("shipping_limit_ts", "shipping_limit_date"),
    ):
        from_order, from_shipment = order.get(key), shipment.get(key)
        merged[key] = from_shipment or from_order
        a, b = to_datetime(from_order), to_datetime(from_shipment)
        if a and b and abs((a - b).total_seconds()) > 24 * 3600:
            cls.conflicts.append(
                {
                    "field": field_name,
                    "sources": ["get_order", "get_shipment_summary"],
                    "selected_source": "get_shipment_summary",
                    "resolution_code": "PREFER_SHIPMENT_TRACKING",
                }
            )
    status_order, status_shipment = order.get("order_status"), shipment.get("status")
    if status_order and status_shipment and status_order != status_shipment:
        cls.conflicts.append(
            {
                "field": "order_status",
                "sources": ["get_order", "get_shipment_summary"],
                "selected_source": "get_order",
                "resolution_code": "PREFER_ORDER_RECORD",
            }
        )
    return merged


def _split_or_duplicate(
    captures: list[dict[str, Any]], totals: list[float]
) -> tuple[str, dict[str, Any]] | None:
    values = [c["value"] for c in captures if c["value"] and c["value"] > 0]
    if len(values) < 2:
        return None
    for size in range(2, min(len(values), 4) + 1):
        for combo in combinations(values, size):
            if any(close(sum(combo), total) for total in totals):
                return "valid_split_payment", {"parts": list(combo)}
    for first, second in combinations(values, 2):
        if close(first, second):
            return "duplicate_charge", {"amount": money(second)}
    return None


def classify(state: CaseState) -> Classification:
    order = state.facts("order-agent")
    payment = state.facts("payment-agent")
    shipment = state.facts("shipment-agent")
    claims = state.hints.get("claims", [])
    cls = Classification(issue="insufficient_evidence", rule="R00_NO_ORDER", claims=claims)
    timeline = _timeline(state, cls)
    cls.details["timeline"] = timeline

    if not order.get("order_found"):
        cls.notes.append("order_evidence_missing")
        return cls

    status = (order.get("order_status") or "").lower()
    paid = payment.get("paid_total")
    refunds = payment.get("refunds") or []
    refunded = sum(r["amount"] or 0 for r in refunds if (r["status"] or "") in REFUND_DONE)
    cls.details.update(paid=paid, total=order.get("order_total"), refunded=round(refunded, 2))

    failed = [r for r in refunds if (r["status"] or "") in REFUND_FAILED]
    pending = [r for r in refunds if (r["status"] or "") in REFUND_PENDING]
    if failed:
        cls.issue, cls.rule = "refund_failed", "R10_REFUND_FAILED_IN_SCOPE"
        cls.details["refund"] = failed[-1]
        return cls
    if pending:
        cls.issue, cls.rule = "refund_pending", "R11_REFUND_PENDING_IN_SCOPE"
        cls.details["refund"] = pending[-1]
        return cls

    if status in CANCELED | UNAVAILABLE:
        if not payment.get("payment_found"):
            cls.notes.append("payment_evidence_missing")
            cls.rule = "R20_STATUS_WITHOUT_PAYMENT"
            return cls
        outstanding = round((paid or 0) - refunded, 2)
        if outstanding > 0.009:
            canceled = status in CANCELED
            cls.issue = "canceled_order_paid" if canceled else "unavailable_order_paid"
            cls.rule = "R21_CANCELED_PAID" if canceled else "R22_UNAVAILABLE_PAID"
            cls.details["outstanding"] = outstanding
            return cls
        cls.issue, cls.rule = "unsupported_claim", "R23_ALREADY_REFUNDED_OR_UNPAID"
        return cls

    mismatches = [
        event
        for event in payment.get("lifecycle") or []
        if "mismatch" in (event["type"] or "") and event["status"] in OPEN_STATUSES
    ]
    if mismatches:
        cls.issue, cls.rule = "payment_mismatch", "R30_RECONCILIATION_MISMATCH_OPEN"
        cls.details["mismatch_amount"] = mismatches[0]["amount"]
        return cls

    delivered = _date(timeline["delivered_customer_ts"])
    estimated = _date(timeline["estimated_ts"])
    if delivered and estimated and delivered > estimated:
        handover = to_datetime(timeline["delivered_carrier_ts"])
        limit = to_datetime(timeline["shipping_limit_ts"])
        cls.details["days_late"] = (delivered - estimated).days
        actors = {
            str(event["actor"] or "").lower()
            for event in shipment.get("events") or []
            if "late" in event["type"]
        }
        if handover and limit:
            seller_fault = handover > limit
            cls.rule = "R40_HANDOVER_AFTER_LIMIT" if seller_fault else "R41_CARRIER_AFTER_HANDOVER"
            expected_actor = "seller" if seller_fault else "logistics_provider"
            if actors and expected_actor not in actors:
                cls.notes.append("late_actor_disagrees")
        else:
            seller_fault = "seller" in actors
            cls.rule = "R42_LATE_ACTOR_FROM_EVENTS"
            cls.notes.append("handover_timestamps_missing")
        cls.issue = "late_delivery_seller" if seller_fault else "late_delivery_logistics"
        return cls

    if not payment.get("payment_found") and {"duplicate", "mismatch"} & set(claims):
        cls.notes.append("payment_evidence_missing")
        cls.rule = "R33_PAYMENT_CLAIM_WITHOUT_LEDGER"
        return cls

    totals = unique([order.get("order_total"), *(order.get("item_totals") or [])])
    found = _split_or_duplicate(payment.get("captures") or [], [t for t in totals if t])
    if found:
        cls.issue, details = found
        cls.rule = (
            "R31_SPLIT_MATCHES_TOTAL"
            if cls.issue == "valid_split_payment"
            else ("R32_EQUAL_CAPTURES_EXCEED_TOTAL")
        )
        cls.details.update(details)
        return cls

    if "late" in claims and not delivered:
        cls.notes.append("delivery_timestamps_missing")
        cls.rule = "R43_LATE_CLAIM_WITHOUT_DELIVERY_DATA"
        return cls

    cls.issue, cls.rule = "unsupported_claim", "R50_NO_ANOMALY_IN_EVIDENCE"
    return cls


def policy_rule(policy: Any, issue: str) -> dict[str, Any]:
    rules = pick(policy, "rules")
    rule = rules.get(issue) if isinstance(rules, dict) else None
    return rule if isinstance(rule, dict) else {}


def _fallback_refund(state: CaseState, cls: Classification) -> float:
    order = state.facts("order-agent")
    issue = cls.issue
    if issue in {"canceled_order_paid", "unavailable_order_paid"}:
        return money(cls.details.get("outstanding"))
    if issue in {"late_delivery_seller", "late_delivery_logistics"}:
        return money(order.get("freight_total"))
    if issue == "duplicate_charge":
        return money(cls.details.get("amount"))
    if issue == "payment_mismatch":
        return money(cls.details.get("mismatch_amount"))
    if issue == "refund_failed":
        return money((cls.details.get("refund") or {}).get("amount"))
    return 0.0


def resolve(state: CaseState, cls: Classification, policy: Any) -> Decision:
    order = state.facts("order-agent")
    issue = cls.issue
    order_id = order.get("order_id")
    sellers = order.get("seller_ids") or []
    rule = policy_rule(policy, issue)
    default = DEFAULT_RULES[issue]
    if issue != "insufficient_evidence" and not rule:
        cls.notes.append("policy_rule_missing")

    status = rule.get("case_status") or default["case_status"]
    action = rule.get("recommended_action") or default["action"]

    parties: list[tuple[str, str | None]] = []
    for raw in rule.get("responsible_parties") or [{"party_type": default["party"]}]:
        party_type = raw.get("party_type") if isinstance(raw, dict) else None
        if party_type not in PARTY_TYPES:
            continue
        if party_type == "seller":
            # The policy names the liable *role*; the seller identity comes from evidence.
            parties.extend(("seller", seller) for seller in sellers[:5])
            if not sellers:
                parties.append(("seller", None))
        else:
            party_id = raw.get("party_id")
            parties.append((party_type, party_id if isinstance(party_id, str) else None))

    amount = to_float(rule.get("refund_brl"))
    if amount is None:
        amount = _fallback_refund(state, cls)
    refund_lines: list[tuple[str, float, str | None]] = []
    if amount and amount > 0 and issue in REFUND_REASON:
        refund_lines.append((REFUND_REASON[issue], money(amount), order_id))

    causes = list(CAUSE_CODES[issue])
    if issue == "late_delivery_seller" and "handover_timestamps_missing" in cls.notes:
        causes[1] = "SELLER_REPORTED_LATE"

    return Decision(
        primary_issue=issue,
        case_status=status,
        confidence=0.0,
        ranked_causes=causes[:5],
        responsible_parties=list(dict.fromkeys(parties))[:5],
        refund_lines=refund_lines,
        resolution_actions=[action[:80]],
        evidence_domains=list(ISSUE_DOMAINS[issue]),
        rule=cls.rule,
        data_conflicts=cls.conflicts[:5],
        confidence_notes=list(cls.notes),
    )
