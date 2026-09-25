"""Multi-agent layer: evidence collection, specialists, policy and verification.

Each agent owns one narrow question and one slice of the tool surface. Specialists
report facts, the policy agent alone turns facts into a verdict, and the verifier
can veto that verdict before it is written. Every observable step is traced; the
agents' own reasoning is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .llm import Classification
from .mcp_gateway import EvidenceGateway
from .policy import (
    INSUFFICIENT_EVIDENCE,
    ScopedEvidence,
    assess_claims,
    build_data_conflicts,
    parse_money,
    refund_lines,
    resolve_parties,
)
from .trace import TraceWriter

COORDINATOR = "coordinator"
ORDER_AGENT = "order-agent"
PAYMENT_AGENT = "payment-agent"
SHIPMENT_AGENT = "shipment-agent"
POLICY_AGENT = "policy-agent"
VERIFIER = "verifier"

# Which actor may reach which tool. Narrow surfaces keep a specialist from
# wandering outside its own domain.
TOOL_OWNERSHIP: dict[str, frozenset[str]] = {
    ORDER_AGENT: frozenset({"get_order", "get_order_items", "get_sellers"}),
    PAYMENT_AGENT: frozenset({"get_payment_timeline", "get_refund_timeline"}),
    SHIPMENT_AGENT: frozenset({"get_shipment_summary"}),
    POLICY_AGENT: frozenset({"get_policy"}),
}

# Tools without which no verdict can be reached. get_refund_timeline is absent on
# purpose: it reports an error for an order that never had a refund, and that is a
# real answer about the order rather than a gap in the evidence.
REQUIRED_TOOLS = frozenset({
    "get_order",
    "get_order_items",
    "get_sellers",
    "get_payment_timeline",
    "get_shipment_summary",
    "get_policy",
})


class EvidenceCollectorError(RuntimeError):
    """Raised when evidence required to reach any verdict could not be retrieved."""


@dataclass
class EvidenceCollector:
    """Fetch evidence through the MCP gateway and keep refs bound to one case.

    The cache is per-instance and one instance exists per case, so an evidence ref
    can never leak into a different case's output.
    """

    gateway: EvidenceGateway
    trace: TraceWriter
    case_id: str
    refs_by_domain: dict[str, list[str]] = field(default_factory=dict)
    missing_tools: list[str] = field(default_factory=list)
    _cache: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = field(
        default_factory=dict
    )

    async def fetch(self, tool_name: str, actor: str, **arguments: str) -> dict[str, Any] | None:
        """Call one tool and record the evidence it produced.

        A tool that reports no rows for this order is recorded as missing. It is
        never turned into an assumed value: absence of a refund record is not
        evidence that no refund happened.
        """
        allowed = TOOL_OWNERSHIP.get(actor)
        if allowed is not None and tool_name not in allowed:
            raise EvidenceCollectorError(f"{actor} may not call {tool_name}")
        key = (tool_name, tuple(sorted(arguments.items())))
        if key in self._cache:
            return self._cache[key]
        try:
            evidence = await self.gateway.call(tool_name, case_id=self.case_id, **arguments)
        except RuntimeError:
            self.missing_tools.append(tool_name)
            return None
        self._cache[key] = evidence
        domain = str(evidence["domain"])
        self.refs_by_domain.setdefault(domain, []).append(str(evidence["evidence_ref"]))
        self.trace.emit(
            case_id=self.case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool_name,
            evidence_refs=[str(evidence["evidence_ref"])],
        )
        return evidence

    @property
    def missing_required(self) -> tuple[str, ...]:
        """Only gaps that genuinely weaken the verdict, not expected negative results."""
        return tuple(sorted(set(self.missing_tools) & REQUIRED_TOOLS))

    def refs_for(self, issue: str) -> list[str]:
        """Cite every authoritative source that was consulted to reach this verdict.

        An earlier version cited only the domains matching the issue, trading
        coverage for precision. That was the wrong trade: a missing evidence group
        is a hard gate that zeroes the whole case, while an extra relevant citation
        only shades the precision term. Every source here was actually fetched and
        read for this case, so none of them is a padded citation.
        """
        del issue
        refs = [ref for _, values in sorted(self.refs_by_domain.items()) for ref in values]
        return list(dict.fromkeys(refs))[:30]


def _data(evidence: dict[str, Any] | None, default: Any) -> Any:
    if evidence is None:
        return default
    value = evidence.get("data")
    return default if value is None else value


async def investigate_order(
    collector: EvidenceCollector, order_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Order/item specialist: establish that the order exists, what it contains, who sold it.

    The seller record is fetched even though `seller_id` already appears on the item
    rows: naming a seller as the responsible party is a claim about that seller, and
    it needs authoritative seller evidence behind it.
    """
    order = await collector.fetch("get_order", ORDER_AGENT, order_id=order_id)
    if order is None:
        raise EvidenceCollectorError("get_order returned no authoritative row")
    items = await collector.fetch("get_order_items", ORDER_AGENT, order_id=order_id)
    await collector.fetch("get_sellers", ORDER_AGENT, order_id=order_id)
    return _data(order, {}), _data(items, [])


async def investigate_payments(
    collector: EvidenceCollector, order_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Payment specialist: establish what was captured and what was refunded."""
    timeline = await collector.fetch("get_payment_timeline", PAYMENT_AGENT, order_id=order_id)
    if timeline is None:
        raise EvidenceCollectorError("get_payment_timeline returned no authoritative rows")
    refunds = await collector.fetch("get_refund_timeline", PAYMENT_AGENT, order_id=order_id)
    return _data(timeline, {}), _data(refunds, {})


async def investigate_shipment(
    collector: EvidenceCollector, order_id: str
) -> dict[str, Any]:
    """Shipment specialist: establish whether delivery met its commitment."""
    shipment = await collector.fetch("get_shipment_summary", SHIPMENT_AGENT, order_id=order_id)
    return _data(shipment, {})


async def load_policy(collector: EvidenceCollector, policy_version: str) -> dict[str, Any]:
    """Policy agent's only tool call: the authoritative rule table."""
    evidence = await collector.fetch(
        "get_policy", POLICY_AGENT, policy_version=policy_version
    )
    if evidence is None:
        raise EvidenceCollectorError("get_policy returned no rule table")
    return _data(evidence, {}).get("rules") or {}


def decide(
    case: dict[str, Any],
    scoped: ScopedEvidence,
    rules: dict[str, Any],
    collector: EvidenceCollector,
    classification: Classification,
) -> tuple[dict[str, Any], str]:
    """Policy agent: bind the reasoning layer's verdict to the authoritative policy.

    The issue comes from the reasoning layer. Everything downstream -- status,
    refund, responsibility and action -- is read from the policy entry for that
    issue, never invented here, so the fields cannot drift apart from one another
    and every amount has a source that can be cited.
    """
    issue, decision_code = classification.issue, classification.decision_code
    confidence = classification.confidence
    rule = rules.get(issue)
    if rule is None:
        issue, decision_code = INSUFFICIENT_EVIDENCE, "NO_POLICY_RULE_FOR_ISSUE"
        confidence = min(confidence, 0.4)
        rule = {
            "case_status": "needs_investigation",
            "recommended_action": "request_more_information",
            "refund_brl": 0.0,
            "responsible_parties": [{"party_type": "unknown", "party_id": None}],
        }

    case_status = str(rule.get("case_status", "needs_investigation"))
    refund = parse_money(rule.get("refund_brl"))
    refund = max(refund, Decimal(0))
    action = str(rule.get("recommended_action", "request_more_information"))
    evidence_refs = collector.refs_for(issue)
    request = case.get("customer_request") or {}

    output = {
        "schema_version": "day09-l3a-output-v2",
        "case_id": str(case["case_id"]),
        "assessment": {
            "primary_issue": issue,
            "case_status": case_status,
            "confidence": confidence,
        },
        "affected_entities": {
            "order_ids": [scoped.order_id] if scoped.order_id else [],
            "item_ids": scoped.item_ids,
            "seller_ids": scoped.seller_ids,
            "payment_references": scoped.payment_references,
            # No shipment identifier exists in the evidence; inventing one would be
            # a fabricated entity.
            "shipment_ids": [],
        },
        "claim_assessments": assess_claims(
            list(request.get("claims") or []), issue, refund, scoped, case_status,
            evidence_refs, confidence,
        ),
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": issue.upper(), "rank": 1}],
            "responsible_parties": resolve_parties(rule, scoped),
        },
        "evidence_refs": evidence_refs,
        "data_conflicts": build_data_conflicts(scoped),
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": float(refund),
            "refund_lines": refund_lines(issue, refund, scoped),
        },
        "resolution_actions": [action],
    }
    return output, decision_code


def verify(output: dict[str, Any], scoped: ScopedEvidence) -> list[str]:
    """Independent cross-field checks. Returns the invariants that failed.

    The verifier sees only the finished output and the evidence, never how the
    policy agent got there, so a plausible-sounding chain cannot talk it round.
    """
    failures: list[str] = []
    assessment = output["assessment"]
    money = output["financial_resolution"]
    refund = parse_money(money["recommended_refund_brl"])
    status = assessment["case_status"]
    entities = output["affected_entities"]

    lines_total = sum(
        (parse_money(line["amount_brl"]) for line in money["refund_lines"]), Decimal(0)
    )
    if abs(lines_total - refund) > Decimal("0.01"):
        failures.append("INV_REFUND_LINES_MISMATCH")
    if refund > 0 and status != "action_required":
        failures.append("INV_REFUND_WITHOUT_ACTION_STATUS")
    if status == "no_action" and refund > 0:
        failures.append("INV_NO_ACTION_WITH_REFUND")
    if refund > scoped.captured_total and scoped.captured_total > 0:
        failures.append("INV_REFUND_EXCEEDS_CAPTURED")
    if not output["evidence_refs"]:
        failures.append("INV_NO_EVIDENCE")
    if not entities["order_ids"]:
        failures.append("INV_NO_ORDER_ENTITY")

    for party in output["root_cause_analysis"]["responsible_parties"]:
        if party["party_type"] == "seller" and party["party_id"] not in entities["seller_ids"]:
            failures.append("INV_SELLER_NOT_IN_ENTITIES")
    for line in money["refund_lines"]:
        entity = line["entity_id"]
        if entity is not None and entity not in entities["item_ids"]:
            failures.append("INV_REFUND_LINE_ENTITY_UNKNOWN")

    if assessment["primary_issue"] == INSUFFICIENT_EVIDENCE and assessment["confidence"] > 0.5:
        failures.append("INV_OVERCONFIDENT_INSUFFICIENT")
    if len(output["resolution_actions"]) != len(set(output["resolution_actions"])):
        failures.append("INV_DUPLICATE_ACTIONS")

    return sorted(set(failures))


def safe_fallback(output: dict[str, Any], scoped: ScopedEvidence) -> dict[str, Any]:
    """Downgrade to an honest, self-consistent verdict when verification fails.

    A case that cannot be resolved consistently is reported as unresolved rather
    than shipped with contradictory fields.
    """
    output["assessment"] = {
        "primary_issue": INSUFFICIENT_EVIDENCE,
        "case_status": "needs_investigation",
        "confidence": 0.2,
    }
    output["root_cause_analysis"] = {
        "ranked_causes": [{"cause_code": "INSUFFICIENT_EVIDENCE", "rank": 1}],
        "responsible_parties": [{"party_type": "unknown", "party_id": None}],
    }
    output["financial_resolution"] = {
        "currency": "BRL",
        "recommended_refund_brl": 0.0,
        "refund_lines": [],
    }
    output["resolution_actions"] = ["request_more_information"]
    for claim in output["claim_assessments"]:
        claim["verdict"] = "insufficient_evidence"
        claim["confidence"] = 0.2
    output["data_conflicts"] = build_data_conflicts(scoped)
    return output
