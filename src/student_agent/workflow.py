"""L3A coordinator: routes one case through the specialist, policy and verifier agents.

Flow (all correlated by ``case_id``)::

    coordinator -> order-agent -> coordinator -> payment-agent -> coordinator
                -> shipment-agent -> coordinator -> policy-agent -> verifier-agent
                -> coordinator (output)
"""

from __future__ import annotations

from typing import Any

from . import OUTPUT_SCHEMA_VERSION
from .evidence import OLIST_ID, pick, strings, text_blob
from .mcp_gateway import EvidenceGateway
from .policy import classify, detect_claims, resolve
from .specialists import OrderItemAgent, PaymentAgent, ShipmentAgent, Specialist
from .state import MAX_HOPS, AgentMessage, CaseState, Decision, Finding
from .trace import TraceWriter
from .verifier import CLAIM_ISSUES, verify

COORDINATOR = "coordinator"
POLICY_AGENT = "policy-agent"
VERIFIER = "verifier-agent"
ISSUE_ARGUMENT_KEYS = (
    "primary_issue",
    "issue",
    "issue_code",
    "issue_type",
    "policy_type",
    "policy_topic",
    "topic",
    "category",
    "claim_type",
    "dispute_type",
)


class PolicyAgent(Specialist):
    name = POLICY_AGENT
    tools = {"policy": ("get_policy", "get_refund_policy", "lookup_policy", "get_policies")}

    async def run(self, state: CaseState, message: AgentMessage) -> Finding:
        finding = Finding(agent=self.name)
        cls = classify(state)
        issue_context = dict.fromkeys(ISSUE_ARGUMENT_KEYS, cls.issue)
        context = {**state.hints, **message.payload.get("context", {}), **issue_context}
        policy = await self.fetch(state, finding, "policy", context, "dispute_policy")
        decision = resolve(state, cls, policy.data if policy else {})
        state.decision = decision
        state.record(finding)
        cited = cited_refs(state, decision)
        self.trace.emit(
            case_id=state.case_id,
            event_type="policy_decided",
            actor=self.name,
            decision_code=decision.primary_issue,
            evidence_refs=cited[:20] or None,
            attributes={
                "rule": decision.rule,
                "case_status": decision.case_status,
                "refund_brl": decision.refund_total,
                "policy_consulted": policy is not None,
            },
        )
        finding.facts = {"rule": decision.rule}
        return finding


def extract_hints(case: dict[str, Any]) -> dict[str, Any]:
    """IDs and claim categories stated in the case input (claims, not facts)."""
    text = text_blob(case)
    hints: dict[str, Any] = {}
    for key in (
        "order_id",
        "customer_id",
        "seller_id",
        "product_id",
        "item_id",
        "payment_reference",
        "payment_id",
        "shipment_id",
        "tracking_id",
        "refund_id",
        "policy_id",
    ):
        value = pick(case, key)
        if isinstance(value, str | int) and str(value).strip():
            hints[key] = str(value).strip()
    if "order_id" not in hints:
        candidates = OLIST_ID.findall(text)
        if candidates:
            hints["order_id"] = candidates[0]
    hints["claims"] = detect_claims(case, text)
    return hints


def cited_refs(state: CaseState, decision: Decision) -> list[str]:
    """Only evidence from the domains that support the conclusion is cited."""
    refs = [item.ref for item in state.ledger.values() if item.domain in decision.evidence_domains]
    if not refs:
        refs = [item.ref for item in state.ledger.values()]
    return list(dict.fromkeys(refs))[:30]


class Coordinator:
    def __init__(self, gateway: EvidenceGateway, trace: TraceWriter) -> None:
        self.gateway = gateway
        self.trace = trace

    def send(self, state: CaseState, message: AgentMessage, event_type: str) -> None:
        state.hops += 1
        if state.hops > MAX_HOPS:
            raise RuntimeError(f"{state.case_id}: A2A hop limit exceeded")
        self.trace.emit(
            case_id=state.case_id,
            event_type=event_type,
            actor=message.sender,
            target=message.recipient,
            decision_code=message.intent,
            evidence_refs=message.payload.get("evidence_refs") or None,
            attributes={"message_id": message.message_id, "hop": state.hops},
        )

    def message(
        self, state: CaseState, sender: str, recipient: str, intent: str, **payload: Any
    ) -> AgentMessage:
        return AgentMessage(state.case_id, sender, recipient, intent, state.hops + 1, payload)

    async def delegate(self, state: CaseState, agent: Specialist, intent: str, **payload: Any):
        assignment = self.message(state, COORDINATOR, agent.name, intent, **payload)
        self.send(state, assignment, "task_assigned")
        finding = await agent.run(state, assignment)
        state.record(finding)
        reply = self.message(
            state,
            agent.name,
            COORDINATOR,
            "FINDINGS_READY" if finding.evidence else "FINDINGS_EMPTY",
            evidence_refs=finding.refs[:20],
        )
        self.send(state, reply, "handoff")
        return finding

    async def solve(self, case: dict[str, Any]) -> dict[str, Any]:
        state = CaseState(case_id=case["case_id"], case=case, hints=extract_hints(case))
        specs = await self.gateway.tool_specs()
        order_agent = OrderItemAgent(self.gateway, self.trace, specs)
        payment_agent = PaymentAgent(self.gateway, self.trace, specs)
        shipment_agent = ShipmentAgent(self.gateway, self.trace, specs)
        policy_agent = PolicyAgent(self.gateway, self.trace, specs)

        order = await self.delegate(state, order_agent, "COLLECT_ORDER_ITEM_SELLER")
        context = {
            "order_id": order.facts.get("order_id") or state.hints.get("order_id"),
            "customer_id": order.facts.get("customer_id"),
        }
        if order.facts.get("payment_references"):
            context["payment_reference"] = order.facts["payment_references"][0]
        if order.facts.get("shipment_ids"):
            context["shipment_id"] = order.facts["shipment_ids"][0]
        context = {key: value for key, value in context.items() if value}
        await self.delegate(
            state, payment_agent, "COLLECT_PAYMENT_REFUND", context=context, need_refund=True
        )
        await self.delegate(state, shipment_agent, "COLLECT_SHIPMENT", context=context)

        to_policy = self.message(
            state,
            COORDINATOR,
            POLICY_AGENT,
            "DECIDE_POLICY",
            context=context,
            evidence_refs=list(state.ledger)[:20],
        )
        self.send(state, to_policy, "handoff")
        await policy_agent.run(state, to_policy)
        decision = state.decision
        assert decision is not None

        to_verifier = self.message(
            state,
            POLICY_AGENT,
            VERIFIER,
            "VERIFY_DECISION",
            evidence_refs=cited_refs(state, decision)[:20],
        )
        self.send(state, to_verifier, "handoff")
        original_issue = decision.primary_issue
        decision, checks = verify(state, decision)
        state.decision = decision
        output = build_output(state, decision)
        self.trace.contracts.validate_output(output, f"outputs/{state.case_id}.json")
        unknown = set(output["evidence_refs"]) - set(state.ledger)
        if unknown:
            raise RuntimeError(f"{state.case_id}: output cites evidence outside this case")
        self.trace.emit(
            case_id=state.case_id,
            event_type="verification_completed",
            actor=VERIFIER,
            decision_code="REVISED" if decision.primary_issue != original_issue else "PASSED",
            evidence_refs=output["evidence_refs"][:20] or None,
            attributes={
                "checks": ",".join(checks)[:200],
                "confidence": decision.confidence,
                "primary_issue": decision.primary_issue,
            },
        )
        back = self.message(state, VERIFIER, COORDINATOR, "VALIDATED_OUTPUT")
        self.send(state, back, "handoff")
        return output


def _claim_assessments(state: CaseState, decision: Decision, refs: list[str]) -> list[dict]:
    claims = state.case.get("claims")
    if not isinstance(claims, list):
        return []
    result = []
    for claim in claims[:5]:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str):
            continue
        categories = detect_claims(claim, text_blob(claim))
        related = set().union(*(CLAIM_ISSUES.get(c, set()) for c in categories))
        if decision.primary_issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
        elif (
            decision.primary_issue == "unsupported_claim"
            or (related and decision.primary_issue not in related)
            or decision.primary_issue == "valid_split_payment"
        ):
            verdict = "unsupported"
        else:
            verdict = "supported"
        result.append(
            {
                "claim_id": claim["claim_id"][:64],
                "verdict": verdict,
                "confidence": decision.confidence,
                "evidence_refs": refs,
            }
        )
    return result


def build_output(state: CaseState, decision: Decision) -> dict[str, Any]:
    order = state.facts("order-agent")
    payment = state.facts("payment-agent")
    shipment = state.facts("shipment-agent")
    refs = cited_refs(state, decision)
    order_ids = strings([order.get("order_id")]) if order.get("order_found") else []
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": state.case_id,
        "assessment": {
            "primary_issue": decision.primary_issue,
            "case_status": decision.case_status,
            "confidence": decision.confidence,
        },
        "affected_entities": {
            "order_ids": order_ids[:20],
            "item_ids": strings(i["item_id"] for i in order.get("items") or [])[:20],
            "seller_ids": strings(order.get("seller_ids") or [])[:20],
            "payment_references": strings(
                [
                    *(order.get("payment_references") or []),
                    *(payment.get("payment_references") or []),
                ]
            )[:20],
            "shipment_ids": strings(
                [*(order.get("shipment_ids") or []), *(shipment.get("shipment_ids") or [])]
            )[:20],
        },
        "root_cause_analysis": {
            "ranked_causes": [
                {"cause_code": code, "rank": rank}
                for rank, code in enumerate(decision.ranked_causes[:5], 1)
            ],
            "responsible_parties": [
                {"party_type": party_type, "party_id": party_id}
                for party_type, party_id in decision.responsible_parties
            ],
        },
        "evidence_refs": refs,
        "data_conflicts": decision.data_conflicts[:5],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": decision.refund_total,
            "refund_lines": [
                {"reason_code": code, "amount_brl": amount, "entity_id": entity}
                for code, amount, entity in decision.refund_lines[:10]
            ],
        },
        "resolution_actions": decision.resolution_actions[:8],
    }
    assessments = _claim_assessments(state, decision, refs)
    if assessments:
        output["claim_assessments"] = assessments
    return output


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Entry point used by ``day09 run``."""
    return await Coordinator(gateway, trace).solve(case)
