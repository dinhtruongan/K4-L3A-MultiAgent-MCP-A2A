"""Measure the local LLM against the deterministic engine on cached evidence.

Runs entirely offline against `evidence-cache.json`, so prompts can be changed and
re-measured without touching MCP. Three arms per case:

  rule      deterministic engine
  llm       the production path: analyst + auditor passes with a tie-break
  llm_raw   one pass over unscoped rows, i.e. no timestamp filtering at all

The third arm is the control. It shows what the model is up against when nothing
removes the rows belonging to other scenarios first.

    python experiments/dump_evidence.py          # once
    python experiments/llm_second_opinion.py --per-topic 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_agent.cases import load_case_set  # noqa: E402
from student_agent.llm import _LABELS, ANALYST_PROMPT, LLMUnavailable, _ask, classify  # noqa: E402
from student_agent.policy import detect_issue, scope_evidence  # noqa: E402

CACHE = ROOT / "experiments" / "evidence-cache.json"


def render_raw_facts(payload: dict) -> str:
    """Every row exactly as the tools returned it, with no scoping applied."""
    order = payload["order"]
    lines = [
        f"order_status: {order.get('order_status')}",
        f"purchased_at: {order.get('order_purchase_timestamp')}",
        f"approved_at: {order.get('order_approved_at')}",
        f"delivered_to_customer_at: {order.get('order_delivered_customer_date')}",
        f"estimated_delivery_at: {order.get('order_estimated_delivery_date')}",
    ]
    lines += [
        f"item: price={row.get('price')} freight={row.get('freight_value')} "
        f"shipping_limit={row.get('shipping_limit_date')}"
        for row in payload["items"]
    ]
    lines += [
        f"payment: seq={row.get('payment_sequential')} value={row.get('payment_value')}"
        for row in payload["payment_timeline"].get("payments") or []
    ]
    lines += [
        f"payment_event: {e.get('event_at')} {e.get('event_type')} "
        f"{e.get('amount_brl')} status={e.get('status')}"
        for e in payload["payment_timeline"].get("events") or []
    ]
    lines += [
        f"refund_event: {e.get('event_at')} {e.get('event_type')} "
        f"{e.get('amount_brl')} status={e.get('status')}"
        for e in (payload["refund_timeline"] or {}).get("events") or []
    ]
    lines += [
        f"shipment_event: {e.get('event_at')} {e.get('event_type')} actor={e.get('actor')}"
        for e in (payload["shipment"] or {}).get("events") or []
    ]
    return "\n".join(lines)


def pick_cases(case_set, per_topic: int) -> list[str]:
    """Even sample across topics so no scenario dominates the score."""
    by_topic: dict[str, list[str]] = defaultdict(list)
    for case_id in case_set.case_ids:
        topics = [
            claim["topic"]
            for claim in case_set.cases[case_id]["customer_request"]["claims"]
            if claim["topic"] != "requested_full_refund"
        ]
        by_topic[topics[0] if topics else "none"].append(case_id)
    chosen: list[str] = []
    for topic in sorted(by_topic):
        chosen.extend(by_topic[topic][:per_topic])
    return sorted(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--per-topic", type=int, default=5)
    parser.add_argument("--skip-raw", action="store_true", help="skip the unscoped arm")
    args = parser.parse_args()

    if not CACHE.exists():
        print("evidence cache missing; run experiments/dump_evidence.py first")
        return 1
    cache = json.loads(CACHE.read_text())
    case_set = load_case_set(ROOT)
    selected = [cid for cid in pick_cases(case_set, args.per_topic) if cid in cache]

    rows = []
    started = time.time()
    for case_id in selected:
        payload = cache[case_id]
        scoped = scope_evidence(
            payload["order"], payload["items"], payload["payment_timeline"],
            payload["refund_timeline"], payload["shipment"],
        )
        topic = [
            claim["topic"]
            for claim in case_set.cases[case_id]["customer_request"]["claims"]
            if claim["topic"] != "requested_full_refund"
        ][0]
        rule, _ = detect_issue(scoped)
        result = classify(scoped, {topic}, args.model)

        raw = "-"
        if not args.skip_raw:
            try:
                raw, _ = _ask(
                    args.model,
                    ANALYST_PROMPT.format(labels=_LABELS, facts=render_raw_facts(payload)),
                )
            except LLMUnavailable:
                raw = "<no answer>"

        rows.append((case_id, topic, rule, result, raw))
        marks = "".join(
            "." if value == topic else "X"
            for value in (rule, result.issue, result.analyst_issue)
        )
        print(f"{case_id} {marks} truth={topic:24s} llm={result.issue:24s} {result.source}")

    total = len(rows)
    rule_ok = sum(1 for _, topic, rule, _, _ in rows if rule == topic)
    prod_ok = sum(1 for _, topic, _, res, _ in rows if res.issue == topic)
    analyst_ok = sum(1 for _, topic, _, res, _ in rows if res.analyst_issue == topic)
    auditor_ok = sum(1 for _, topic, _, res, _ in rows if res.auditor_issue == topic)
    raw_ok = sum(1 for _, topic, _, _, raw in rows if raw == topic)
    agreed = sum(1 for _, _, _, res, _ in rows if res.agreed)

    print(f"\n{'=' * 74}")
    print(f"cases                                  : {total}")
    print(f"deterministic engine                   : {rule_ok}/{total}")
    print(f"LLM analyst pass   (scoped evidence)   : {analyst_ok}/{total}")
    print(f"LLM auditor pass   (scoped evidence)   : {auditor_ok}/{total}")
    print(f"LLM production    (2 pass + tie-break) : {prod_ok}/{total}")
    if not args.skip_raw:
        print(f"LLM single pass    (UNSCOPED rows)     : {raw_ok}/{total}   <- control arm")
    print(f"two passes agreed with each other      : {agreed}/{total}")
    print(f"wall clock                             : {time.time() - started:.0f}s")

    confusion = Counter(
        (topic, res.issue) for _, topic, _, res, _ in rows if res.issue != topic
    )
    if confusion:
        print("\nproduction-path mistakes:")
        for (truth, got), count in confusion.most_common():
            print(f"  {count:3d}  {truth}  ->  {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
