from __future__ import annotations

from typing import Any

from student_agent.agents import verify
from student_agent.policy import (
    CANCELED_ORDER_PAID,
    DUPLICATE_CHARGE,
    LATE_DELIVERY_LOGISTICS,
    LATE_DELIVERY_SELLER,
    PAYMENT_MISMATCH,
    REFUND_FAILED,
    REFUND_PENDING,
    UNAVAILABLE_ORDER_PAID,
    UNSUPPORTED_CLAIM,
    VALID_SPLIT_PAYMENT,
    detect_issue,
    scope_evidence,
    score_confidence,
)

PURCHASE = "2018-05-25T09:00:00-03:00"
APPROVED = "2018-05-25T10:00:00-03:00"
LIMIT = "2018-05-28T09:00:00-03:00"
DELIVERED = "2018-06-03T09:00:00-03:00"
ESTIMATED = "2018-06-04T09:00:00-03:00"
FAR_AWAY = "2018-09-06T09:00:00-03:00"


def order(status: str = "delivered", delivered: str | None = DELIVERED) -> dict[str, Any]:
    return {
        "order_id": "order-1",
        "order_status": status,
        "order_purchase_timestamp": PURCHASE,
        "order_approved_at": APPROVED,
        "order_delivered_customer_date": delivered,
        "order_estimated_delivery_date": ESTIMATED,
    }


def item(freight: str = "10.00", limit: str = LIMIT) -> dict[str, Any]:
    return {
        "order_item_id": "item-1",
        "seller_id": "seller-1",
        "price": "79.00",
        "freight_value": freight,
        "shipping_limit_date": limit,
    }


def capture(amount: str, at: str = APPROVED, kind: str = "captured") -> dict[str, Any]:
    return {"event_at": at, "event_type": kind, "amount_brl": amount, "status": "confirmed"}


def pay(amount: str, sequential: str = "1", kind: str = "credit_card") -> dict[str, Any]:
    return {
        "payment_sequential": sequential,
        "payment_type": kind,
        "payment_installments": "1",
        "payment_value": amount,
    }


def build(
    status: str = "delivered",
    delivered: str | None = DELIVERED,
    items: list[dict[str, Any]] | None = None,
    payments: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    refunds: list[dict[str, Any]] | None = None,
    shipments: list[dict[str, Any]] | None = None,
):
    return scope_evidence(
        order=order(status, delivered),
        items=items if items is not None else [item()],
        payment_timeline={
            "payments": payments if payments is not None else [pay("89.00")],
            "events": events if events is not None else [capture("89.00")],
        },
        refund_timeline={"events": refunds or []},
        shipment={"events": shipments or []},
    )


def test_scoping_drops_rows_from_other_scenarios() -> None:
    scoped = build(
        items=[item(), item(freight="18.00", limit=FAR_AWAY)],
        payments=[pay("89.00"), pay("16.00")],
        events=[capture("89.00"), capture("16.00", at=FAR_AWAY)],
        shipments=[{"event_at": FAR_AWAY, "event_type": "delivered_late", "actor": "seller"}],
    )
    assert len(scoped.items) == 1
    assert len(scoped.payment_events) == 1
    assert scoped.shipment_events == []
    assert scoped.dropped == {"items": 1, "payment_events": 1, "shipment_events": 1}
    # The out-of-window capture must not inflate the money the case is judged on.
    assert str(scoped.captured_total) == "89.00"


def test_canceled_and_unavailable_orders_are_separated() -> None:
    assert detect_issue(build(status="canceled", delivered=None))[0] == CANCELED_ORDER_PAID
    assert detect_issue(build(status="unavailable", delivered=None))[0] == UNAVAILABLE_ORDER_PAID


def test_refund_lifecycle_outranks_other_signals() -> None:
    failed = build(refunds=[{"event_at": DELIVERED, "amount_brl": "52.00", "status": "failed"}])
    pending = build(refunds=[{"event_at": DELIVERED, "amount_brl": "89.00", "status": "pending"}])
    assert detect_issue(failed)[0] == REFUND_FAILED
    assert detect_issue(pending)[0] == REFUND_PENDING


def test_payment_mismatch_needs_an_explicit_reconciliation_event() -> None:
    scoped = build(
        payments=[pay("35.00")],
        events=[capture("35.00"), capture("35.00", kind="reconciliation_mismatch")],
    )
    assert detect_issue(scoped)[0] == PAYMENT_MISMATCH


def test_duplicate_charge_is_distinguished_from_valid_split_payment() -> None:
    duplicate = build(
        payments=[pay("64.00"), pay("64.00", sequential="2", kind="voucher")],
        events=[capture("64.00"), capture("64.00")],
    )
    split = build(
        payments=[pay("44.50"), pay("44.50", sequential="2", kind="voucher")],
        events=[capture("44.50"), capture("44.50")],
    )
    # Same shape, two sequentials, equal amounts. Only the total tells them apart.
    assert detect_issue(duplicate)[0] == DUPLICATE_CHARGE
    assert detect_issue(split)[0] == VALID_SPLIT_PAYMENT


def test_late_delivery_is_attributed_to_the_actor_in_the_event() -> None:
    # Delivered 2018-06-06, two days past the 2018-06-04 estimate.
    late = "2018-06-06T09:00:00-03:00"
    seller = build(
        delivered=late,
        shipments=[{"event_at": late, "event_type": "delivered_late", "actor": "seller"}],
    )
    carrier = build(
        delivered=late,
        shipments=[
            {"event_at": late, "event_type": "delivered_late", "actor": "logistics_provider"}
        ],
    )
    assert detect_issue(seller)[0] == LATE_DELIVERY_SELLER
    assert detect_issue(carrier)[0] == LATE_DELIVERY_LOGISTICS


def test_clean_order_yields_unsupported_claim() -> None:
    assert detect_issue(build())[0] == UNSUPPORTED_CLAIM


def test_late_event_is_rejected_when_the_order_arrived_on_time() -> None:
    # An on-time order cannot have been delivered late. A stray `delivered_late`
    # row landing near the delivery date belongs to another scenario, so the
    # order's own timestamps have to win over it.
    on_time = build(
        delivered="2018-06-03T09:00:00-03:00",  # estimate is 2018-06-04
        shipments=[
            {
                "event_at": "2018-05-31T09:00:00-03:00",
                "event_type": "delivered_late",
                "actor": "logistics_provider",
            }
        ],
    )
    assert on_time.shipment_events == []
    assert on_time.dropped.get("shipment_events") == 1
    assert detect_issue(on_time)[0] == UNSUPPORTED_CLAIM


def test_confidence_is_never_certain_and_drops_when_the_claim_is_contradicted() -> None:
    scoped = build()
    agreeing = score_confidence(UNSUPPORTED_CLAIM, {UNSUPPORTED_CLAIM}, scoped)
    disagreeing = score_confidence(UNSUPPORTED_CLAIM, {CANCELED_ORDER_PAID}, scoped)
    assert agreeing < 1.0
    assert disagreeing < agreeing


def test_verifier_rejects_a_refund_that_contradicts_its_own_status() -> None:
    scoped = build()
    output = {
        "assessment": {
            "primary_issue": VALID_SPLIT_PAYMENT,
            "case_status": "no_action",
            "confidence": 0.9,
        },
        "affected_entities": {
            "order_ids": ["order-1"],
            "item_ids": ["item-1"],
            "seller_ids": ["seller-1"],
            "payment_references": ["1"],
            "shipment_ids": [],
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": "VALID_SPLIT_PAYMENT", "rank": 1}],
            "responsible_parties": [{"party_type": "customer", "party_id": None}],
        },
        "evidence_refs": ["ev_" + "a" * 24],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 89.0,
            "refund_lines": [],
        },
        "resolution_actions": ["document_no_action"],
    }
    failures = verify(output, scoped)
    assert "INV_NO_ACTION_WITH_REFUND" in failures
    assert "INV_REFUND_LINES_MISMATCH" in failures
