"""Deterministic policy engine for L3A.

Money and responsibility decisions never touch an LLM: every conclusion here is a
pure function of scoped MCP evidence plus the authoritative policy table returned
by ``get_policy``. That keeps the same case reproducible across runs and makes the
rules unit-testable without the network.

Case inputs carry two overlapping layers of rows: the order's own lifecycle and
unrelated rows from other scenarios. Everything below works on the scoped layer,
selected by :func:`scope_evidence`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

CANCELED_ORDER_PAID = "canceled_order_paid"
UNAVAILABLE_ORDER_PAID = "unavailable_order_paid"
LATE_DELIVERY_SELLER = "late_delivery_seller"
LATE_DELIVERY_LOGISTICS = "late_delivery_logistics"
VALID_SPLIT_PAYMENT = "valid_split_payment"
PAYMENT_MISMATCH = "payment_mismatch"
DUPLICATE_CHARGE = "duplicate_charge"
REFUND_PENDING = "refund_pending"
REFUND_FAILED = "refund_failed"
UNSUPPORTED_CLAIM = "unsupported_claim"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

# How far a row may sit from its anchor and still belong to this order.
ITEM_LIMIT_WINDOW = timedelta(days=7)
PAYMENT_EVENT_WINDOW = timedelta(days=3)
SHIPMENT_EVENT_WINDOW = timedelta(days=3)
REFUND_TAIL_WINDOW = timedelta(days=14)

def parse_time(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, returning None for absent or malformed values."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_money(value: Any) -> Decimal:
    """Parse a BRL amount. Missing or malformed amounts count as zero, never as a guess."""
    if isinstance(value, int | float | Decimal):
        return Decimal(str(value))
    if isinstance(value, str) and value.strip():
        try:
            return Decimal(value.strip())
        except InvalidOperation:
            return Decimal(0)
    return Decimal(0)


def _within(moment: datetime | None, anchor: datetime | None, window: timedelta) -> bool:
    return moment is not None and anchor is not None and abs(moment - anchor) <= window


@dataclass(frozen=True)
class ScopedEvidence:
    """The subset of every MCP payload that provably belongs to this order."""

    order_id: str
    order_status: str
    purchased_at: datetime | None
    approved_at: datetime | None
    delivered_to_customer_at: datetime | None
    estimated_delivery_at: datetime | None
    items: list[dict[str, Any]] = field(default_factory=list)
    payments: list[dict[str, Any]] = field(default_factory=list)
    payment_events: list[dict[str, Any]] = field(default_factory=list)
    refund_events: list[dict[str, Any]] = field(default_factory=list)
    shipment_events: list[dict[str, Any]] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)
    missing_tools: tuple[str, ...] = ()

    @property
    def item_total(self) -> Decimal:
        return sum(
            (parse_money(row.get("price")) + parse_money(row.get("freight_value"))
             for row in self.items),
            Decimal(0),
        )

    @property
    def captured_total(self) -> Decimal:
        return sum(
            (parse_money(event.get("amount_brl"))
             for event in self.payment_events
             if event.get("event_type") == "captured"),
            Decimal(0),
        )

    @property
    def seller_ids(self) -> list[str]:
        return sorted({str(row["seller_id"]) for row in self.items if row.get("seller_id")})

    @property
    def item_ids(self) -> list[str]:
        return sorted({str(row["order_item_id"]) for row in self.items if row.get("order_item_id")})

    @property
    def payment_references(self) -> list[str]:
        return sorted(
            {str(row["payment_sequential"]) for row in self.payments
             if row.get("payment_sequential")}
        )

    @property
    def delivered_late(self) -> bool:
        return (
            self.delivered_to_customer_at is not None
            and self.estimated_delivery_at is not None
            and self.delivered_to_customer_at > self.estimated_delivery_at
        )


def scope_evidence(
    order: dict[str, Any],
    items: list[dict[str, Any]],
    payment_timeline: dict[str, Any],
    refund_timeline: dict[str, Any] | None,
    shipment: dict[str, Any] | None,
    missing_tools: tuple[str, ...] = (),
) -> ScopedEvidence:
    """Keep only the rows anchored to this order's own lifecycle.

    Each source carries rows from unrelated scenarios. They are dropped here rather
    than reasoned about later, and the count of dropped rows is reported so the
    caller can declare the conflict instead of hiding it.
    """
    purchased_at = parse_time(order.get("order_purchase_timestamp"))
    approved_at = parse_time(order.get("order_approved_at")) or purchased_at
    delivered_at = parse_time(order.get("order_delivered_customer_date"))
    estimated_at = parse_time(order.get("order_estimated_delivery_date"))
    delivery_anchor = delivered_at or estimated_at

    kept_items = [
        row for row in items
        if purchased_at is None
        or _within(parse_time(row.get("shipping_limit_date")), purchased_at, ITEM_LIMIT_WINDOW)
    ]

    events = payment_timeline.get("events") or []
    kept_payment_events = [
        event for event in events
        if _within(parse_time(event.get("event_at")), approved_at, PAYMENT_EVENT_WINDOW)
    ]

    # Payment rows carry no timestamp of their own; they are matched by the amounts
    # that the scoped capture events confirm.
    scoped_amounts = [parse_money(event.get("amount_brl")) for event in kept_payment_events]
    kept_payments = []
    remaining = list(scoped_amounts)
    for row in payment_timeline.get("payments") or []:
        amount = parse_money(row.get("payment_value"))
        if amount in remaining:
            remaining.remove(amount)
            kept_payments.append(row)

    refund_rows = (refund_timeline or {}).get("events") or []
    kept_refunds = [
        event for event in refund_rows
        if _in_refund_window(parse_time(event.get("event_at")), purchased_at, delivery_anchor)
    ]

    # An order delivered on or before its estimate cannot have been delivered late.
    # A `delivered_late` event that contradicts the order's own timestamps belongs to
    # another scenario however close it lands, so the order timeline wins.
    delivered_late = (
        delivered_at is not None and estimated_at is not None and delivered_at > estimated_at
    )
    shipment_rows = (shipment or {}).get("events") or []
    kept_shipments = [
        event for event in shipment_rows
        if _within(parse_time(event.get("event_at")), delivery_anchor, SHIPMENT_EVENT_WINDOW)
        and (event.get("event_type") != "delivered_late" or delivered_late)
    ]

    dropped = {
        "items": len(items) - len(kept_items),
        "payment_events": len(events) - len(kept_payment_events),
        "refund_events": len(refund_rows) - len(kept_refunds),
        "shipment_events": len(shipment_rows) - len(kept_shipments),
    }
    return ScopedEvidence(
        order_id=str(order.get("order_id", "")),
        order_status=str(order.get("order_status", "")),
        purchased_at=purchased_at,
        approved_at=approved_at,
        delivered_to_customer_at=delivered_at,
        estimated_delivery_at=estimated_at,
        items=kept_items,
        payments=kept_payments,
        payment_events=kept_payment_events,
        refund_events=kept_refunds,
        shipment_events=kept_shipments,
        dropped={key: value for key, value in dropped.items() if value > 0},
        missing_tools=missing_tools,
    )


def _in_refund_window(
    moment: datetime | None, purchased_at: datetime | None, delivery_anchor: datetime | None
) -> bool:
    if moment is None or purchased_at is None:
        return False
    end = (delivery_anchor or purchased_at) + REFUND_TAIL_WINDOW
    return purchased_at <= moment <= end


def detect_issue(scoped: ScopedEvidence) -> tuple[str, str]:
    """Classify the order from scoped evidence alone.

    Returns the primary issue plus the decision code naming the rule that fired.
    Rules are ordered most specific first; the customer's own claim is never an
    input here, so a false claim cannot steer the verdict.
    """
    if not scoped.order_status:
        return INSUFFICIENT_EVIDENCE, "NO_ORDER_EVIDENCE"

    captured = scoped.captured_total
    if scoped.order_status == "canceled" and captured > 0:
        return CANCELED_ORDER_PAID, "ORDER_CANCELED_WITH_CAPTURE"
    if scoped.order_status == "unavailable" and captured > 0:
        return UNAVAILABLE_ORDER_PAID, "ORDER_UNAVAILABLE_WITH_CAPTURE"

    refund_states = {str(event.get("status", "")) for event in scoped.refund_events}
    if "failed" in refund_states:
        return REFUND_FAILED, "REFUND_LIFECYCLE_FAILED"
    if "pending" in refund_states:
        return REFUND_PENDING, "REFUND_LIFECYCLE_PENDING"

    if any(event.get("event_type") == "reconciliation_mismatch"
           for event in scoped.payment_events):
        return PAYMENT_MISMATCH, "PAYMENT_RECONCILIATION_MISMATCH"

    captures = [event for event in scoped.payment_events if event.get("event_type") == "captured"]
    amounts = [parse_money(event.get("amount_brl")) for event in captures]
    total = scoped.item_total
    if len(amounts) > 1 and len(set(amounts)) == 1 and captured > total:
        return DUPLICATE_CHARGE, "DUPLICATE_CAPTURE_ABOVE_ORDER_TOTAL"

    for event in scoped.shipment_events:
        if event.get("event_type") != "delivered_late":
            continue
        if event.get("actor") == "seller":
            return LATE_DELIVERY_SELLER, "SHIPMENT_LATE_SELLER_HANDOFF"
        if event.get("actor") == "logistics_provider":
            return LATE_DELIVERY_LOGISTICS, "SHIPMENT_LATE_CARRIER"

    if len(scoped.payment_references) > 1 and captured == total and total > 0:
        return VALID_SPLIT_PAYMENT, "SPLIT_PAYMENT_RECONCILES_TO_TOTAL"

    if not scoped.payment_events and not scoped.items:
        return INSUFFICIENT_EVIDENCE, "NO_SCOPED_EVIDENCE"

    return UNSUPPORTED_CLAIM, "NO_ANOMALY_IN_SCOPED_EVIDENCE"


def resolve_parties(
    rule: dict[str, Any], scoped: ScopedEvidence
) -> list[dict[str, Any]]:
    """Take responsibility from policy, but bind seller identity to this order.

    The policy table is shared by every case, so its seller ``party_id`` is a
    template value. Naming a seller that does not appear in this order's items
    would contradict ``affected_entities``, so the scoped seller is used instead.
    """
    parties: list[dict[str, Any]] = []
    for entry in rule.get("responsible_parties") or []:
        party_type = str(entry.get("party_type", "unknown"))
        party_id = entry.get("party_id")
        if party_type == "seller":
            sellers = scoped.seller_ids
            party_id = sellers[0] if sellers else None
        parties.append({"party_type": party_type, "party_id": party_id})
    return parties or [{"party_type": "unknown", "party_id": None}]


def refund_lines(issue: str, amount: Decimal, scoped: ScopedEvidence) -> list[dict[str, Any]]:
    """Split the policy refund into one attributable line, or none when nothing is owed."""
    if amount <= 0:
        return []
    entity_id = scoped.item_ids[0] if scoped.item_ids else None
    return [{
        "reason_code": issue.upper(),
        "amount_brl": float(amount),
        "entity_id": entity_id,
    }]


def build_data_conflicts(scoped: ScopedEvidence) -> list[dict[str, Any]]:
    """Declare every source row that was dropped, rather than silently discarding it."""
    labels = {
        "items": "order_items.shipping_limit_date",
        "payment_events": "payment_timeline.events",
        "refund_events": "refund_timeline.events",
        "shipment_events": "shipment_summary.events",
    }
    conflicts = []
    for key, count in sorted(scoped.dropped.items()):
        conflicts.append({
            "field": labels.get(key, key),
            "sources": ["order_lifecycle_window", "out_of_window_rows"],
            "selected_source": "order_lifecycle_window",
            "resolution_code": f"SCOPED_BY_ORDER_WINDOW_DROPPED_{count}",
        })
    return conflicts[:5]


def score_confidence(issue: str, claimed_topics: set[str], scoped: ScopedEvidence) -> float:
    """Report how well the evidence pins the verdict, never a flat 1.0.

    Confidence tracks evidence quality so that a wrong verdict is also a hedged one:
    corroboration from the customer's claim raises it, a contradicted claim or a
    failed tool lowers it.
    """
    confidence = 0.55
    if issue in claimed_topics:
        confidence += 0.20
    else:
        confidence -= 0.15
    if not scoped.missing_tools:
        confidence += 0.10
    else:
        confidence -= 0.20
    if scoped.payment_events and scoped.items:
        confidence += 0.05
    if issue == INSUFFICIENT_EVIDENCE:
        confidence -= 0.25
    return round(min(0.95, max(0.05, confidence)), 2)


# How decisive each verdict is, as a fraction of the case-level confidence. A
# clear yes or no carries the full weight; a hedged verdict carries less.
VERDICT_CERTAINTY = {
    "supported": 1.0,
    "unsupported": 1.0,
    "partially_supported": 0.9,
    "insufficient_evidence": 0.45,
}


def assess_claims(
    claims: list[dict[str, Any]], issue: str, refund: Decimal, scoped: ScopedEvidence,
    case_status: str, evidence_refs: list[str], case_confidence: float,
) -> list[dict[str, Any]]:
    """Judge each customer claim against the evidence, including claims that fail."""
    total = scoped.item_total
    assessments = []
    for claim in claims[:5]:
        topic = str(claim.get("topic", ""))
        if topic == "requested_full_refund":
            if case_status == "needs_investigation":
                verdict = "insufficient_evidence"
            elif refund <= 0:
                verdict = "unsupported"
            elif total > 0 and refund >= total:
                verdict = "supported"
            else:
                verdict = "partially_supported"
        elif topic == issue:
            verdict = "supported"
        else:
            verdict = "unsupported"
        confidence = round(case_confidence * VERDICT_CERTAINTY[verdict], 2)
        assessments.append({
            "claim_id": str(claim.get("claim_id", "")),
            "verdict": verdict,
            "confidence": min(0.95, max(0.05, confidence)),
            "evidence_refs": evidence_refs,
        })
    return assessments
