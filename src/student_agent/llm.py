"""LLM reasoning layer: classify a case from scoped evidence.

The model is asked exactly one question -- which issue the evidence shows -- and
is never asked how much money to refund or who pays. Amounts and responsibility
come from the authoritative policy table fetched over MCP, because a model that
invents a refund figure has no source to cite for it.

Two passes run per case with different framings: the policy agent classifies, and
the verifier classifies again without seeing the first pass's reasoning. Agreement
raises confidence; disagreement lowers it and hands the tie to the deterministic
engine. If the model is unreachable or answers outside the allowed labels, the
deterministic engine decides alone rather than the run failing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen

from .policy import INSUFFICIENT_EVIDENCE, ScopedEvidence, detect_issue, parse_money

# Override with LLM_MODEL in .env to try a different local model.
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.1:latest")

ISSUES = (
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
    "payment_mismatch",
    "duplicate_charge",
    "refund_pending",
    "refund_failed",
    "unsupported_claim",
    "insufficient_evidence",
)

_LABELS = "\n".join(f"- {name}" for name in ISSUES)

# Two framings of the same question. The verifier is told to re-derive the answer
# rather than review one, so it cannot be anchored by a first pass it never sees.
ANALYST_PROMPT = """You are a dispute analyst for a Brazilian e-commerce platform.
Decide which single issue the evidence below demonstrates.

Rules:
- Use only the evidence. The customer's own description is a claim, not a fact.
- valid_split_payment requires AT LEAST TWO distinct payment sequentials and a
  captured total equal to the order total. One payment is never a split payment.
- A late-delivery label requires both a late delivery timestamp and a matching
  delivered_late event naming the responsible actor.
- A refund label requires a refund event; duplicate_charge requires two captures
  above the order total; payment_mismatch requires a reconciliation_mismatch event.
- If the evidence shows nothing wrong with the order, answer unsupported_claim.
- If the evidence is too thin to tell, answer insufficient_evidence.

Allowed labels (choose exactly one):
{labels}

Evidence:
{facts}

Answer as JSON: {{"primary_issue": "<one label>", "why": "<max 15 words>"}}"""

AUDITOR_PROMPT = """You are an independent auditor. Working only from the record
below, determine what actually happened to this order. Do not assume any earlier
conclusion is correct.

Checklist:
- An order status of canceled or unavailable with money captured is a paid-but-\
undelivered case.
- A refund event decides refund_pending vs refund_failed.
- A reconciliation_mismatch event means the captured amount does not reconcile.
- Two identical captures above the order total is a duplicate charge.
- Several payment sequentials that sum to the order total is a valid split.
- Exactly one payment sequential is not a split. A delivered-on-time order with
  one matching capture and no refund or late event is unsupported_claim.
- A delivered_late event names the party at fault.
- No anomaly at all means the claim is unsupported.

Allowed labels (choose exactly one):
{labels}

Record:
{facts}

Answer as JSON: {{"primary_issue": "<one label>", "why": "<max 15 words>"}}"""


class LLMUnavailable(RuntimeError):
    """The local model could not be reached or gave no usable label."""


@dataclass(frozen=True)
class Classification:
    """What the reasoning layer concluded, and how much to trust it."""

    issue: str
    decision_code: str
    confidence: float
    source: str
    analyst_issue: str
    auditor_issue: str
    rationale: str

    @property
    def agreed(self) -> bool:
        return self.analyst_issue == self.auditor_issue


def render_facts(scoped: ScopedEvidence) -> str:
    """Render scoped evidence as plain facts, one per line.

    Only rows that survived scoping appear. Rows belonging to other scenarios are
    already gone, so the model is not asked to do the filtering that timestamps
    can settle exactly.
    """
    lines = [
        f"order_status: {scoped.order_status}",
        f"delivered_to_customer_at: {scoped.delivered_to_customer_at}",
        f"estimated_delivery_at: {scoped.estimated_delivery_at}",
        f"delivered_after_estimate: {scoped.delivered_late}",
        f"order_total_brl: {scoped.item_total}",
        f"captured_total_brl: {scoped.captured_total}",
        f"distinct_payment_sequentials: {scoped.payment_references}",
    ]
    lines += [
        f"payment_event: {event.get('event_type')} amount={event.get('amount_brl')} "
        f"status={event.get('status')}"
        for event in scoped.payment_events
    ]
    lines += [
        f"refund_event: {event.get('event_type')} amount={event.get('amount_brl')} "
        f"status={event.get('status')}"
        for event in scoped.refund_events
    ]
    lines += [
        f"shipment_event: {event.get('event_type')} actor={event.get('actor')}"
        for event in scoped.shipment_events
    ]
    if not scoped.refund_events:
        lines.append("refund_event: none on record")
    if not scoped.shipment_events:
        lines.append("shipment_event: none on record")
    return "\n".join(lines)


def _supported_by_evidence(issue: str, scoped: ScopedEvidence) -> bool:
    """Reject a model label when its required factual signal is absent."""
    captures = [
        event for event in scoped.payment_events if event.get("event_type") == "captured"
    ]
    refund_states = {str(event.get("status", "")) for event in scoped.refund_events}
    late_actors = {
        str(event.get("actor", "")) for event in scoped.shipment_events
        if event.get("event_type") == "delivered_late"
    }
    if issue == "canceled_order_paid":
        return scoped.order_status == "canceled" and scoped.captured_total > 0
    if issue == "unavailable_order_paid":
        return scoped.order_status == "unavailable" and scoped.captured_total > 0
    if issue == "late_delivery_seller":
        return scoped.delivered_late and "seller" in late_actors
    if issue == "late_delivery_logistics":
        return scoped.delivered_late and "logistics_provider" in late_actors
    if issue == "valid_split_payment":
        return (
            len(scoped.payment_references) >= 2
            and scoped.item_total > 0
            and scoped.captured_total == scoped.item_total
        )
    if issue == "payment_mismatch":
        return any(
            event.get("event_type") == "reconciliation_mismatch"
            for event in scoped.payment_events
        )
    if issue == "duplicate_charge":
        amounts = [parse_money(event.get("amount_brl")) for event in captures]
        return len(amounts) >= 2 and len(set(amounts)) == 1 and sum(amounts) > scoped.item_total
    if issue == "refund_pending":
        return "pending" in refund_states
    if issue == "refund_failed":
        return "failed" in refund_states
    if issue == "unsupported_claim":
        return detect_issue(scoped)[0] == issue
    return True


def _ask(model: str, prompt: str) -> tuple[str, str]:
    """One deterministic call. Raises LLMUnavailable rather than guessing a label."""
    try:
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "num_predict": 120, "seed": 7},
        }).encode("utf-8")
        request = Request(
            os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
        answer = json.loads(result["message"]["content"])
    except Exception as exc:  # transport, decode and server faults all mean "no answer"
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
    issue = str(answer.get("primary_issue", "")).strip()
    if issue not in ISSUES:
        raise LLMUnavailable(f"model returned an unknown label: {issue[:40]!r}")
    return issue, str(answer.get("why", "")).strip()[:120]


def classify(
    scoped: ScopedEvidence, claimed_topics: set[str], model: str = DEFAULT_MODEL
) -> Classification:
    """Classify one case with two independent LLM passes and a deterministic tie-break.

    Confidence reflects how much the two passes and the corroborating claim line
    up, so a case the model is genuinely unsure about is reported as unsure.
    """
    fallback_issue, fallback_code = detect_issue(scoped)
    facts = render_facts(scoped)

    try:
        analyst, rationale = _ask(model, ANALYST_PROMPT.format(labels=_LABELS, facts=facts))
    except LLMUnavailable as exc:
        return Classification(
            issue=fallback_issue,
            decision_code=f"DETERMINISTIC_{fallback_code}",
            confidence=_confidence(fallback_issue, claimed_topics, scoped, 0.55),
            source=f"deterministic-fallback ({exc})"[:80],
            analyst_issue="",
            auditor_issue="",
            rationale="",
        )

    try:
        auditor, _ = _ask(model, AUDITOR_PROMPT.format(labels=_LABELS, facts=facts))
    except LLMUnavailable:
        auditor = ""

    if auditor and analyst == auditor:
        issue, source, base = analyst, "llm-consensus", 0.90
    elif auditor:
        # The two passes split. The deterministic engine breaks the tie, and the
        # disagreement is carried into confidence rather than hidden.
        issue, source, base = fallback_issue, "llm-split-deterministic-tiebreak", 0.55
    else:
        issue, source, base = analyst, "llm-single-pass", 0.75

    if not _supported_by_evidence(issue, scoped):
        issue, source, base = fallback_issue, "llm-label-rejected-by-evidence", 0.60

    return Classification(
        issue=issue,
        decision_code=f"LLM_{issue.upper()}",
        confidence=_confidence(issue, claimed_topics, scoped, base),
        source=source,
        analyst_issue=analyst,
        auditor_issue=auditor,
        rationale=rationale,
    )


def _confidence(
    issue: str, claimed_topics: set[str], scoped: ScopedEvidence, base: float
) -> float:
    """Adjust a base confidence by corroboration and evidence completeness."""
    confidence = base
    confidence += 0.05 if issue in claimed_topics else -0.15
    if scoped.missing_tools:
        confidence -= 0.20
    if not (scoped.payment_events and scoped.items):
        confidence -= 0.10
    if issue == INSUFFICIENT_EVIDENCE:
        confidence -= 0.25
    return round(min(0.95, max(0.05, confidence)), 2)
