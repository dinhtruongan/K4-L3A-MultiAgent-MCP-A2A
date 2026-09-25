"""Fetch every case's raw tool payloads once and cache them on disk.

The MCP session drops periodically, and re-fetching evidence for every prompt
tweak is both slow and pointless. This pulls each case once, retries on a dropped
session, and writes one JSON file the offline experiments read.

    python experiments/dump_evidence.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_agent.agents import (  # noqa: E402
    EvidenceCollector,
    investigate_order,
    investigate_payments,
    investigate_shipment,
)
from student_agent.cases import load_case_set  # noqa: E402
from student_agent.config import Settings  # noqa: E402
from student_agent.contracts import Contracts  # noqa: E402
from student_agent.mcp_gateway import connect_gateway  # noqa: E402
from student_agent.trace import TraceWriter  # noqa: E402

CACHE = ROOT / "experiments" / "evidence-cache.json"
MAX_STALLED_SESSIONS = 5


async def drain(pending: list[str], case_set, contracts, trace, settings, out: dict) -> None:
    """Fetch the remaining cases over one session, popping each as it lands."""
    async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gw:
        while pending:
            case_id = pending[0]
            order_id = case_set.cases[case_id]["customer_request"]["claimed_order_id"]
            collector = EvidenceCollector(gateway=gw, trace=trace, case_id=case_id)
            order, items = await investigate_order(collector, order_id)
            payments, refunds = await investigate_payments(collector, order_id)
            shipment = await investigate_shipment(collector, order_id)
            out[case_id] = {
                "order": order,
                "items": items,
                "payment_timeline": payments,
                "refund_timeline": refunds,
                "shipment": shipment,
            }
            pending.pop(0)
            if len(out) % 10 == 0:
                print(f"  {len(out)} cases cached", file=sys.stderr)


async def main() -> int:
    settings = Settings.load(ROOT)
    contracts = Contracts(ROOT / "contracts" / "schemas")
    case_set = load_case_set(ROOT)
    trace = TraceWriter(ROOT / "experiments" / "experiment-trace.jsonl", contracts)

    cached: dict[str, dict] = {}
    pending = list(case_set.case_ids)
    stalled = 0
    while pending:
        before = len(pending)
        try:
            await drain(pending, case_set, contracts, trace, settings, cached)
        except Exception as exc:
            stalled = stalled + 1 if len(pending) == before else 0
            if stalled > MAX_STALLED_SESSIONS:
                raise
            print(
                f"WARN: session lost ({type(exc).__name__}); reconnecting, "
                f"{len(pending)} left",
                file=sys.stderr,
            )

    CACHE.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    print(f"cached {len(cached)} cases -> {CACHE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
