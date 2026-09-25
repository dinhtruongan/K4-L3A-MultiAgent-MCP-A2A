"""Deterministic policy engine used by the policy agent.

``classify`` turns specialist facts into a candidate issue; ``resolve`` applies the
refund/responsibility/action rules, optionally refined by the ``get_policy`` evidence.
The customer message only decides *which* area to look at first; it is never treated as
ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .evidence import close, money, pick, to_datetime, to_float, unique
from .state import CaseState, Decision

CANCELED = {"canceled", "cancelled", "canceled_by_seller", "canceled_by_customer"}
UNAVAILABLE = {"unavailable", "out_of_stock"}
REFUND_PENDING = {"pending", "processing", "requested", "in_progress", "created", "open"}
REFUND_FAILED = {"failed", "rejected", "declined", "error", "reversed", "chargeback_failed"}
REFUND_DONE = {"completed", "succeeded", "success", "refunded", "processed", "paid", "done"}

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
        r"never (arrived|received)",
        r"tr[ễe]",
        r"ch[ậa]m",
        r"demor",
        r"ch[ưu]a nh[ậa]n",
        r"prazo",
    ),
    "duplicate": (
        r"twice",
        r"duplicat",
        r"double",
        r"duas vezes",
        r"2 vezes",
        r"hai l[ầa]n",
        r"cobrad[oa]s? (em )?dobro",
        r"charged two",
    ),
    "mismatch": (
        r"wrong amount",
        r"valor errado",
        r"overcharg",
        r"a mais",
        r"more than",
        r"sai s[ốo] ti[ềe]n",
        r"different amount",
        r"diverg",
        r"incorrect (amount|charge)",
    ),
    "refund": (r"refund", r"reembols", r"estorn", r"ho[àa]n ti[ềe]n", r"devolu"),
}

PAYMENT_CLAIMS = {"duplicate", "mismatch"}

ISSUE_DOMAINS: dict[str, tuple[str, ...]] = {
    "canceled_order_paid": ("order", "payment", "refund", "policy"),
    "unavailable_order_paid": ("order", "item", "payment", "refund", "policy"),
    "late_delivery_seller": ("order", "item", "shipment", "seller", "policy"),
    "late_delivery_logistics": ("order", "item", "shipment", "policy"),
    "valid_split_payment": ("order", "item", "payment", "policy"),
    "payment_mismatch": ("order", "item", "payment", "policy"),
    "duplicate_charge": ("order", "item", "payment", "policy"),
    "refund_pending": ("order", "payment", "refund", "policy"),
    "refund_failed": ("order", "payment", "refund", "policy"),
    "unsupported_claim": ("order", "policy"),
    "insufficient_evidence": ("order", "item", "payment", "shipment", "refund"),
}

UNSUPPORTED_DOMAINS: dict[str, tuple[str, ...]] = {
    "late": ("order", "shipment", "policy"),
    "duplicate": ("order", "item", "payment", "policy"),
    "mismatch": ("order", "item", "payment", "policy"),
    "cancel": ("order", "payment", "policy"),
    "unavailable": ("order", "item", "policy"),
    "refund": ("order", "payment", "refund", "policy"),
}

CAUSE_CODES: dict[str, list[str]] = {
    "canceled_order_paid": ["ORDER_CANCELED_AFTER_PAYMENT", "REFUND_NOT_ISSUED"],
    "unavailable_order_paid": ["ITEM_UNAVAILABLE_AFTER_PAYMENT", "REFUND_NOT_ISSUED"],
    "late_delivery_seller": ["SELLER_LATE_HANDOVER", "DELIVERY_AFTER_ESTIMATE"],
    "late_delivery_logistics": ["CARRIER_TRANSIT_DELAY", "DELIVERY_AFTER_ESTIMATE"],
    "valid_split_payment": ["SPLIT_PAYMENT_MATCHES_ORDER_TOTAL"],
    "payment_mismatch": ["PAYMENT_AMOUNT_MISMATCH"],
    "duplicate_charge": ["DUPLICATE_PAYMENT_CAPTURE"],
    "refund_pending": ["REFUND_PENDING_PROCESSING"],
    "refund_failed": ["REFUND_PROCESSING_FAILED"],
    "unsupported_claim": ["CLAIM_CONTRADICTED_BY_EVIDENCE"],
    "insufficient_evidence": ["MISSING_AUTHORITATIVE_EVIDENCE"],
}

ACTIONS: dict[str, list[str]] = {
    "canceled_order_paid": ["refund_full_payment", "notify_customer"],
    "unavailable_order_paid": ["refund_full_payment", "notify_customer", "flag_seller_inventory"],
    "late_delivery_seller": ["refund_late_delivery_compensation", "notify_customer", "warn_seller"],
    "late_delivery_logistics": [
        "refund_late_delivery_compensation",
        "notify_customer",
        "escalate_to_logistics_provider",
    ],
    "valid_split_payment": ["explain_split_payment_to_customer", "close_case"],
    "payment_mismatch": ["refund_overcharge", "notify_customer", "escalate_to_payment_provider"],
    "duplicate_charge": [
        "refund_duplicate_charge",
        "notify_customer",
        "escalate_to_payment_provider",
    ],
    "refund_pending": ["expedite_pending_refund", "notify_customer"],
    "refund_failed": ["retry_refund", "notify_customer", "escalate_to_payment_provider"],
    "unsupported_claim": ["explain_evidence_to_customer", "close_case"],
    "insufficient_evidence": ["request_additional_information", "escalate_to_human_review"],
}

REFUND_REASON: dict[str, str] = {
    "canceled_order_paid": "CANCELED_ORDER_REFUND",
    "unavailable_order_paid": "UNAVAILABLE_ITEM_REFUND",
    "late_delivery_seller": "LATE_DELIVERY_COMPENSATION",
    "late_delivery_logistics": "LATE_DELIVERY_COMPENSATION",
    "payment_mismatch": "OVERCHARGE_REFUND",
    "duplicate_charge": "DUPLICATE_CHARGE_REFUND",
    "refund_pending": "PENDING_REFUND",
    "refund_failed": "FAILED_REFUND_REISSUE",
}

REFUND_ACTION_PREFIX = "refund_"
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


def detect_claims(case: dict[str, Any], text: str) -> list[str]:
    lowered = text.lower()
    found = [
        claim
        for claim, patterns in CLAIM_PATTERNS.items()
        if any(re.search(pattern, lowered) for pattern in patterns)
    ]
    for key in ("claim_type", "claim_types", "complaint_type", "category", "issue_type"):
        value = case.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, str):
                continue
            for claim, patterns in CLAIM_PATTERNS.items():
                if any(re.search(pattern, item.lower()) for pattern in patterns):
                    found.append(claim)
    return unique(found)


def _date(value: str | None):
    parsed = to_datetime(value)
    return parsed.date() if parsed else None


def _timeline(state: CaseState, cls: Classification) -> dict[str, Any]:
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
        if from_order and from_shipment:
            a, b = to_datetime(from_order), to_datetime(from_shipment)
            if a and b and abs((a - b).total_seconds()) > 24 * 3600:
                cls.conflicts.append(
                    {
                        "field": field_name,
                        "sources": ["order", "shipment"],
                        "selected_source": "shipment",
                        "resolution_code": "PREFER_SHIPMENT_TRACKING",
                    }
                )
    return merged


def _duplicate(payments: list[dict[str, Any]], order_total: float | None) -> dict[str, Any] | None:
    active = [p for p in payments if p["status"] not in {"failed", "declined", "voided"}]
    paid = sum(p["value"] for p in active)
    for index, first in enumerate(active):
        for second in active[index + 1 :]:
            if not close(first["value"], second["value"]) or first["value"] <= 0:
                continue
            if first["type"] and second["type"] and first["type"] != second["type"]:
                continue
            if order_total is not None and not close(paid - first["value"], order_total, 0.05):
                continue
            return {"amount": money(second["value"]), "reference": second["reference"]}
    return None


def classify(state: CaseState) -> Classification:
    order = state.facts("order-agent")
    payment = state.facts("payment-agent")
    claims = state.hints.get("claims", [])
    cls = Classification(issue="insufficient_evidence", rule="R00_NO_ORDER", claims=claims)
    timeline = _timeline(state, cls)
    cls.details["timeline"] = timeline

    if not order.get("order_found"):
        cls.notes.append("order_evidence_missing")
        return cls

    status = (order.get("order_status") or "").lower()
    paid = payment.get("paid_total")
    total = order.get("order_total")
    payments = payment.get("payments") or []
    refunds = payment.get("refunds") or []
    refunded = sum(r["amount"] or 0 for r in refunds if (r["status"] or "") in REFUND_DONE)
    cls.details.update(paid=paid, total=total, refunded=round(refunded, 2))

    claimed_amount = to_float(
        pick(
            state.case, "claimed_amount", "claimed_amount_brl", "amount_claimed", "disputed_amount"
        )
    )
    if claimed_amount is not None and paid is not None and not close(claimed_amount, paid):
        cls.conflicts.append(
            {
                "field": "payment_amount",
                "sources": ["customer_message", "payment"],
                "selected_source": "payment",
                "resolution_code": "PREFER_PAYMENT_LEDGER",
            }
        )

    failed = [r for r in refunds if (r["status"] or "") in REFUND_FAILED]
    pending = [r for r in refunds if (r["status"] or "") in REFUND_PENDING]
    if failed:
        cls.issue, cls.rule = "refund_failed", "R10_REFUND_FAILED"
        cls.details["refund"] = failed[0]
        return cls
    if pending:
        cls.issue, cls.rule = "refund_pending", "R11_REFUND_PENDING"
        cls.details["refund"] = pending[0]
        return cls

    if status in CANCELED | UNAVAILABLE:
        if paid is None:
            cls.notes.append("payment_evidence_missing")
            cls.rule = "R20_STATUS_WITHOUT_PAYMENT"
            return cls
        outstanding = round(paid - refunded, 2)
        if outstanding > 0.009:
            cls.issue = "canceled_order_paid" if status in CANCELED else "unavailable_order_paid"
            cls.rule = "R21_CANCELED_PAID" if status in CANCELED else "R22_UNAVAILABLE_PAID"
            cls.details["outstanding"] = outstanding
            return cls
        cls.issue, cls.rule = "unsupported_claim", "R23_ALREADY_REFUNDED_OR_UNPAID"
        return cls

    if payments and total is not None:
        duplicate = _duplicate(payments, total)
        if duplicate:
            cls.issue, cls.rule = "duplicate_charge", "R30_DUPLICATE_CAPTURE"
            cls.details["duplicate"] = duplicate
            return cls
        if paid is not None and not close(paid, total):
            cls.issue, cls.rule = "payment_mismatch", "R31_PAID_NE_TOTAL"
            cls.details["difference"] = round(paid - total, 2)
            return cls
        split = len(payments) > 1 and paid is not None and close(paid, total)
        if split and (set(claims) & PAYMENT_CLAIMS or not claims):
            cls.issue, cls.rule = "valid_split_payment", "R32_SPLIT_MATCHES_TOTAL"
            return cls
    elif set(claims) & PAYMENT_CLAIMS and not payments:
        cls.notes.append("payment_evidence_missing")
        cls.rule = "R33_PAYMENT_CLAIM_WITHOUT_LEDGER"
        return cls

    delivered = _date(timeline["delivered_customer_ts"])
    estimated = _date(timeline["estimated_ts"])
    if delivered and estimated and delivered > estimated:
        handover = to_datetime(timeline["delivered_carrier_ts"])
        limit = to_datetime(timeline["shipping_limit_ts"])
        cls.details["days_late"] = (delivered - estimated).days
        if handover and limit and handover > limit:
            cls.issue, cls.rule = "late_delivery_seller", "R40_SELLER_MISSED_SHIPPING_LIMIT"
        elif handover and limit:
            cls.issue, cls.rule = "late_delivery_logistics", "R41_CARRIER_EXCEEDED_ESTIMATE"
        else:
            cls.issue, cls.rule = "late_delivery_logistics", "R42_LATE_HANDOVER_UNKNOWN"
            cls.notes.append("handover_timestamps_missing")
        return cls

    if "late" in claims and not delivered:
        cls.notes.append("delivery_timestamps_missing")
        cls.rule = "R43_LATE_CLAIM_WITHOUT_DELIVERY_DATA"
        return cls

    if claims:
        cls.issue, cls.rule = "unsupported_claim", "R50_CLAIM_NOT_SUPPORTED"
        return cls
    cls.rule = "R99_NO_CLAIM_NO_ANOMALY"
    return cls


def _policy_number(policy: Any, *names: str) -> float | None:
    return to_float(pick(policy, *names))


def _policy_strings(policy: Any, *names: str) -> list[str]:
    value = pick(policy, *names)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item[:80] for item in value if item.strip()][:8]
    return []


def late_compensation(state: CaseState, policy: Any) -> float:
    order = state.facts("order-agent")
    freight = order.get("freight_total") or 0.0
    total = order.get("order_total") or 0.0
    rate = _policy_number(
        policy, "refund_percent", "compensation_percent", "refund_rate", "compensation_rate"
    )
    fixed = _policy_number(policy, "refund_amount", "compensation_amount", "fixed_compensation")
    basis = str(pick(policy, "refund_basis", "compensation_basis", "basis") or "").lower()
    if fixed is not None:
        amount = fixed
    elif rate is not None:
        rate = rate / 100 if rate > 1 else rate
        base = total if "total" in basis or "order" in basis else freight
        if "item" in basis or "price" in basis:
            base = total - freight
        amount = base * rate
    else:
        amount = freight
    cap = _policy_number(policy, "max_refund", "refund_cap", "max_compensation")
    if cap is not None:
        amount = min(amount, cap)
    return money(max(amount, 0.0))


def resolve(state: CaseState, cls: Classification, policy: Any) -> Decision:
    order = state.facts("order-agent")
    shipment = state.facts("shipment-agent")
    issue = cls.issue
    order_id = order.get("order_id")
    sellers = order.get("seller_ids") or []

    parties: list[tuple[str, str | None]] = []
    if issue in {"late_delivery_seller", "unavailable_order_paid"}:
        parties = [("seller", seller) for seller in sellers[:5]] or [("seller", None)]
    elif issue == "canceled_order_paid":
        actor = str(order.get("canceled_by") or "").lower()
        if "seller" in actor:
            parties = [("seller", seller) for seller in sellers[:5]] or [("seller", None)]
        else:
            parties = [("platform", None)]
    elif issue == "late_delivery_logistics":
        parties = [("logistics_provider", shipment.get("carrier_id") or shipment.get("carrier"))]
    elif issue in {"payment_mismatch", "duplicate_charge", "refund_failed"}:
        parties = [("payment_provider", None)]
    elif issue == "refund_pending":
        parties = [("platform", None)]
    elif issue == "insufficient_evidence":
        parties = [("unknown", None)]

    policy_party = pick(policy, "responsible_party", "responsible_party_type", "liable_party")
    if (
        isinstance(policy_party, str)
        and policy_party in PARTY_TYPES
        and policy_party != "seller"
        and parties
        and parties[0][0] != policy_party
    ):
        parties = [(policy_party, None)]
        cls.notes.append("responsibility_from_policy")

    refund_lines: list[tuple[str, float, str | None]] = []
    reason = REFUND_REASON.get(issue)
    if issue in {"canceled_order_paid", "unavailable_order_paid"}:
        refund_lines.append((reason, money(cls.details.get("outstanding")), order_id))
    elif issue in {"late_delivery_seller", "late_delivery_logistics"}:
        amount = late_compensation(state, policy)
        if amount > 0:
            refund_lines.append((reason, amount, order_id))
    elif issue == "duplicate_charge":
        duplicate = cls.details["duplicate"]
        refund_lines.append((reason, duplicate["amount"], duplicate["reference"] or order_id))
    elif issue == "payment_mismatch":
        difference = cls.details.get("difference") or 0.0
        if difference > 0:
            refund_lines.append((reason, money(difference), order_id))
    elif issue in {"refund_pending", "refund_failed"}:
        refund = cls.details.get("refund") or {}
        amount = refund.get("amount")
        if amount is None:
            amount = cls.details.get("paid") or 0.0
        refund_lines.append((reason, money(amount), refund.get("refund_id") or order_id))
    refund_lines = [line for line in refund_lines if line[1] > 0]

    if issue == "insufficient_evidence":
        status = "needs_investigation"
    elif issue in {"valid_split_payment", "unsupported_claim"}:
        status = "no_action"
    elif issue == "payment_mismatch" and not refund_lines:
        status = "needs_investigation"
    else:
        status = "action_required"

    actions = list(ACTIONS[issue])
    policy_actions = _policy_strings(policy, "resolution_actions", "required_actions", "actions")
    if policy_actions:
        actions = policy_actions
        cls.notes.append("actions_from_policy")
    if not refund_lines:
        actions = [action for action in actions if not action.startswith(REFUND_ACTION_PREFIX)]
    if status == "no_action":
        actions = [action for action in actions if "escalate" not in action]

    causes = list(CAUSE_CODES[issue])
    policy_cause = pick(policy, "cause_code", "root_cause_code")
    if isinstance(policy_cause, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,79}", policy_cause):
        causes = unique([policy_cause, *causes])

    domains = ISSUE_DOMAINS[issue]
    if issue == "unsupported_claim" and cls.claims:
        domains = tuple(
            unique(d for claim in cls.claims for d in UNSUPPORTED_DOMAINS.get(claim, ()))
        )
        domains = domains or ISSUE_DOMAINS[issue]

    return Decision(
        primary_issue=issue,
        case_status=status,
        confidence=0.0,
        ranked_causes=causes[:5],
        responsible_parties=parties,
        refund_lines=refund_lines,
        resolution_actions=unique(actions)[:8],
        evidence_domains=list(domains),
        rule=cls.rule,
        data_conflicts=cls.conflicts[:5],
        confidence_notes=list(cls.notes),
    )
