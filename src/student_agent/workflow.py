"""L3A Multi-Agent Workflow — sử dụng model < 10B tham số."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .llm_client import ask_llm, parse_json_safe
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────

PRIMARY_ISSUES = [
    "canceled_order_paid", "unavailable_order_paid",
    "late_delivery_seller", "late_delivery_logistics",
    "valid_split_payment", "payment_mismatch",
    "duplicate_charge", "refund_pending", "refund_failed",
    "unsupported_claim", "insufficient_evidence",
]

CASE_STATUSES = ["action_required", "no_action", "needs_investigation"]

PARTY_TYPES = [
    "seller", "platform", "logistics_provider",
    "payment_provider", "customer", "unknown",
]

VERDICTS = ["supported", "unsupported", "partially_supported", "insufficient_evidence"]


# ──────────────────────────────────────────────
#  COORDINATOR: Parse customer complaint
# ──────────────────────────────────────────────

async def coordinator_parse(case: dict[str, Any]) -> dict[str, Any]:
    """Coordinator phân tích customer message, trích xuất IDs và claims."""
    case_id = case["case_id"]
    customer_req = case.get("customer_request", {})
    message = customer_req.get("message", case.get("customer_message", case.get("message", "")))
    claimed_order_id = customer_req.get("claimed_order_id") or case.get("order_id")

    order_ids = [claimed_order_id] if claimed_order_id else []

    system_prompt = """You are a complaint analysis coordinator for an e-commerce platform.
Analyze the customer complaint and extract key information.
Return JSON with these fields:
{
  "summary": "brief summary of the complaint",
  "complaint_type": "delivery_issue|payment_issue|refund_issue|order_issue|other",
  "order_ids": ["list of order IDs mentioned"],
  "claims": ["list of specific claims the customer is making"],
  "urgency": "low|medium|high"
}
Important: Only include IDs explicitly mentioned. Do NOT invent any IDs."""

    user_prompt = f"Case ID: {case_id}\nCustomer message: {message}\nCase data: {json.dumps(case, ensure_ascii=False, default=str)}"

    result = parse_json_safe(await ask_llm(system_prompt, user_prompt))
    llm_order_ids = result.get("order_ids", [])
    all_order_ids = list(dict.fromkeys(order_ids + [oid for oid in llm_order_ids if oid]))
    result["order_ids"] = all_order_ids
    return result


# ──────────────────────────────────────────────
#  SPECIALIST AGENTS
# ──────────────────────────────────────────────

async def order_agent(case_id: str, order_ids: list[str], gateway: EvidenceGateway, trace: TraceWriter, available_tools: list[str]) -> dict[str, Any]:
    """Order Agent: lấy thông tin đơn hàng, items, và sellers."""
    results: dict[str, Any] = {"orders": [], "items": [], "sellers": [], "evidence_refs": []}
    for order_id in order_ids:
        if "get_order" in available_tools:
            try:
                ev = await gateway.call("get_order", case_id=case_id, order_id=order_id)
                results["orders"].append(ev["data"])
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="order-agent", tool_name="get_order",
                           evidence_refs=[ev["evidence_ref"]])
            except (Exception, BaseExceptionGroup) as e:
                logger.debug(f"get_order failed for {case_id}/{order_id}: {e}")

        if "get_order_items" in available_tools:
            try:
                ev = await gateway.call("get_order_items", case_id=case_id, order_id=order_id)
                data = ev["data"]
                if isinstance(data, list):
                    results["items"].extend(data)
                else:
                    results["items"].append(data)
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="order-agent", tool_name="get_order_items",
                           evidence_refs=[ev["evidence_ref"]])
            except (Exception, BaseExceptionGroup) as e:
                logger.debug(f"get_order_items failed for {case_id}/{order_id}: {e}")

        if "get_sellers" in available_tools:
            try:
                ev = await gateway.call("get_sellers", case_id=case_id, order_id=order_id)
                data = ev["data"]
                if isinstance(data, list):
                    results["sellers"].extend(data)
                else:
                    results["sellers"].append(data)
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="order-agent", tool_name="get_sellers",
                           evidence_refs=[ev["evidence_ref"]])
            except (Exception, BaseExceptionGroup) as e:
                logger.debug(f"get_sellers failed for {case_id}/{order_id}: {e}")

    return results


async def payment_agent(case_id: str, order_ids: list[str], gateway: EvidenceGateway, trace: TraceWriter, available_tools: list[str]) -> dict[str, Any]:
    """Payment Agent: lấy thông tin thanh toán và dòng thời gian hoàn tiền."""
    results: dict[str, Any] = {"payments": [], "payment_timeline": [], "refund_timeline": [], "evidence_refs": []}
    for order_id in order_ids:
        if "get_order_payments" in available_tools:
            try:
                ev = await gateway.call("get_order_payments", case_id=case_id, order_id=order_id)
                data = ev["data"]
                if isinstance(data, list):
                    results["payments"].extend(data)
                else:
                    results["payments"].append(data)
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="payment-agent", tool_name="get_order_payments",
                           evidence_refs=[ev["evidence_ref"]])
            except (Exception, BaseExceptionGroup) as e:
                logger.debug(f"get_order_payments failed for {case_id}/{order_id}: {e}")

        if "get_payment_timeline" in available_tools:
            try:
                ev = await gateway.call("get_payment_timeline", case_id=case_id, order_id=order_id)
                data = ev["data"]
                if isinstance(data, list):
                    results["payment_timeline"].extend(data)
                else:
                    results["payment_timeline"].append(data)
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="payment-agent", tool_name="get_payment_timeline",
                           evidence_refs=[ev["evidence_ref"]])
            except (Exception, BaseExceptionGroup) as e:
                logger.debug(f"get_payment_timeline failed for {case_id}/{order_id}: {e}")

        if "get_refund_timeline" in available_tools:
            try:
                ev = await gateway.call("get_refund_timeline", case_id=case_id, order_id=order_id)
                data = ev["data"]
                if isinstance(data, list):
                    results["refund_timeline"].extend(data)
                else:
                    results["refund_timeline"].append(data)
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="payment-agent", tool_name="get_refund_timeline",
                           evidence_refs=[ev["evidence_ref"]])
            except (Exception, BaseExceptionGroup) as e:
                logger.debug(f"get_refund_timeline failed for {case_id}/{order_id}: {e}")

    return results


async def shipment_agent(case_id: str, order_ids: list[str], gateway: EvidenceGateway, trace: TraceWriter, available_tools: list[str]) -> dict[str, Any]:
    """Shipment Agent: lấy thông tin vận chuyển."""
    results: dict[str, Any] = {"shipments": [], "evidence_refs": []}
    for order_id in order_ids:
        if "get_shipment_summary" in available_tools:
            try:
                ev = await gateway.call("get_shipment_summary", case_id=case_id, order_id=order_id)
                results["shipments"].append(ev["data"])
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="shipment-agent", tool_name="get_shipment_summary",
                           evidence_refs=[ev["evidence_ref"]])
            except (Exception, BaseExceptionGroup) as e:
                logger.debug(f"get_shipment_summary failed for {case_id}/{order_id}: {e}")
    return results


async def policy_agent(case_id: str, policy_version: str, gateway: EvidenceGateway, trace: TraceWriter, available_tools: list[str]) -> dict[str, Any]:
    """Policy Agent: lấy thông tin chính sách theo policy_version."""
    results: dict[str, Any] = {"policies": [], "evidence_refs": []}
    if "get_policy" in available_tools:
        try:
            ev = await gateway.call("get_policy", case_id=case_id, policy_version=policy_version)
            results["policies"].append(ev["data"])
            results["evidence_refs"].append(ev["evidence_ref"])
            trace.emit(case_id=case_id, event_type="tool_result_consumed",
                       actor="policy-agent", tool_name="get_policy",
                       evidence_refs=[ev["evidence_ref"]])
        except (Exception, BaseExceptionGroup) as e:
            logger.debug(f"get_policy failed for {case_id}/{policy_version}: {e}")
    return results


# ──────────────────────────────────────────────
#  ANALYZER: Phân tích tổng hợp bằng LLM
# ──────────────────────────────────────────────

async def analyze_case(case: dict[str, Any], parsed: dict[str, Any], order_data: dict[str, Any],
                       payment_data: dict[str, Any], shipment_data: dict[str, Any],
                       policy_data: dict[str, Any], all_evidence_refs: list[str],
                       raw_claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Dùng LLM để phân tích và đưa ra kết luận."""

    system_prompt = f"""You are an expert e-commerce dispute investigation agent.
Analyze the customer complaint and all available evidence to determine the truth.
Return valid JSON with:
{{
  "primary_issue": "<one of: {', '.join(PRIMARY_ISSUES)}>",
  "case_status": "<one of: {', '.join(CASE_STATUSES)}>",
  "confidence": <float 0.0-1.0>,
  "ranked_causes": [{{"cause_code": "<UPPERCASE_CODE>", "rank": 1}}],
  "responsible_parties": [{{"party_type": "<one of: {', '.join(PARTY_TYPES)}>", "party_id": "<id or null>"}}],
  "recommended_refund_brl": <number >= 0>,
  "refund_lines": [{{"reason_code": "<reason>", "amount_brl": <number>, "entity_id": "<id or null>"}}],
  "resolution_actions": ["<action 1>", "<action 2>"],
  "claim_assessments": [{{"claim_id": "<exact claim_id from input>", "verdict": "<one of: {', '.join(VERDICTS)}>", "confidence": <float 0.0-1.0>}}],
  "data_conflicts": []
}}
Rules:
- primary_issue must accurately reflect the root problem from the evidence.
- cause_code must be uppercase alphanumeric (e.g. ORDER_CANCELED_PAID, LATE_DELIVERY_LOGISTICS, PAYMENT_DUPLICATED).
- Each claim in claim_assessments MUST use the EXACT claim_id from the input claims list.
- If evidence is genuinely missing or cannot be retrieved, set primary_issue="insufficient_evidence", case_status="needs_investigation", confidence=0.0.
- Otherwise provide realistic confidence (0.7-0.95) and responsible party."""

    user_prompt = f"""Case ID: {case.get('case_id')}
Customer Request: {json.dumps(case.get('customer_request', {}), ensure_ascii=False)}
Input Claims: {json.dumps(raw_claims, ensure_ascii=False)}
Coordinator Summary: {json.dumps(parsed, ensure_ascii=False)}

=== EVIDENCE FROM MCP GATEWAY ===
Orders: {json.dumps(order_data.get('orders', []), ensure_ascii=False, default=str)}
Items: {json.dumps(order_data.get('items', []), ensure_ascii=False, default=str)}
Sellers: {json.dumps(order_data.get('sellers', []), ensure_ascii=False, default=str)}
Payments: {json.dumps(payment_data.get('payments', []), ensure_ascii=False, default=str)}
Payment Timeline: {json.dumps(payment_data.get('payment_timeline', []), ensure_ascii=False, default=str)}
Refund Timeline: {json.dumps(payment_data.get('refund_timeline', []), ensure_ascii=False, default=str)}
Shipments: {json.dumps(shipment_data.get('shipments', []), ensure_ascii=False, default=str)}
Policies: {json.dumps(policy_data.get('policies', []), ensure_ascii=False, default=str)}
Evidence Refs: {json.dumps(all_evidence_refs)}"""

    return parse_json_safe(await ask_llm(system_prompt, user_prompt))


# ──────────────────────────────────────────────
#  VERIFIER: Kiểm tra và sửa output cho đúng schema
# ──────────────────────────────────────────────

def verify_and_fix(analysis: dict[str, Any], case_id: str, all_evidence_refs: list[str],
                   affected_entities: dict[str, list[str]], raw_claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Verifier: validate và fix output cho đúng schema."""
    # primary_issue
    pi = analysis.get("primary_issue", "insufficient_evidence")
    if pi not in PRIMARY_ISSUES:
        pi = "insufficient_evidence"

    # case_status
    cs = analysis.get("case_status", "needs_investigation")
    if cs not in CASE_STATUSES:
        cs = "needs_investigation"

    # confidence
    conf = analysis.get("confidence", 0.5)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (ValueError, TypeError):
        conf = 0.5

    # ranked_causes
    ranked = []
    for c in analysis.get("ranked_causes", []):
        code = str(c.get("cause_code", "UNKNOWN_CAUSE")).strip()
        if not re.match(r"^[A-Z][A-Z0-9_]{2,79}$", code):
            code = re.sub(r"[^A-Z0-9_]", "_", code.upper())
            if not code or not code[0].isalpha():
                code = "X" + code
            if len(code) < 3:
                code = code + "_XX"
            code = code[:80]
        rank = c.get("rank", len(ranked) + 1)
        try:
            rank = max(1, min(5, int(rank)))
        except (ValueError, TypeError):
            rank = len(ranked) + 1
        ranked.append({"cause_code": code, "rank": rank})
    if not ranked:
        ranked = [{"cause_code": "UNKNOWN_CAUSE", "rank": 1}]
    ranked = ranked[:5]

    # responsible_parties
    parties = []
    for p in analysis.get("responsible_parties", []):
        pt = p.get("party_type", "unknown")
        if pt not in PARTY_TYPES:
            pt = "unknown"
        pid = p.get("party_id")
        if pid is not None:
            pid = str(pid)[:128]
        parties.append({"party_type": pt, "party_id": pid})
    if not parties:
        parties = [{"party_type": "unknown", "party_id": None}]
    parties = parties[:5]

    # financial
    refund = analysis.get("recommended_refund_brl", 0.0)
    try:
        refund = max(0.0, float(refund))
    except (ValueError, TypeError):
        refund = 0.0

    refund_lines = []
    for line in analysis.get("refund_lines", []):
        amt = line.get("amount_brl", 0.0)
        try:
            amt = max(0.0, float(amt))
        except (ValueError, TypeError):
            amt = 0.0
        reason = str(line.get("reason_code", "refund"))[:80] or "refund"
        eid = line.get("entity_id")
        if eid is not None:
            eid = str(eid)[:128]
        refund_lines.append({"reason_code": reason, "amount_brl": amt, "entity_id": eid})
    refund_lines = refund_lines[:10]
    if refund > 0 and not refund_lines:
        refund_lines = [{"reason_code": pi, "amount_brl": refund,
                         "entity_id": affected_entities["order_ids"][0] if affected_entities["order_ids"] else None}]

    # resolution_actions
    actions = [str(a)[:80] for a in analysis.get("resolution_actions", []) if a]
    if not actions:
        actions = ["Review case details and verify records"]
    actions = actions[:8]

    # claim_assessments: Map with actual input claim_ids
    input_claim_ids = [c["claim_id"] for c in raw_claims if "claim_id" in c]
    analysis_claims = {str(cl.get("claim_id")): cl for cl in analysis.get("claim_assessments", [])}

    claims = []
    for cid in input_claim_ids[:5]:
        cl_data = analysis_claims.get(cid, {})
        v = cl_data.get("verdict", "supported" if all_evidence_refs else "insufficient_evidence")
        if v not in VERDICTS:
            v = "insufficient_evidence"
        c = cl_data.get("confidence", 0.8 if all_evidence_refs else 0.0)
        try:
            c = max(0.0, min(1.0, float(c)))
        except (ValueError, TypeError):
            c = 0.5
        claims.append({
            "claim_id": cid[:64],
            "verdict": v,
            "confidence": round(c, 4),
            "evidence_refs": list(dict.fromkeys(all_evidence_refs))[:30],
        })

    # If no input claim_ids, build from analysis
    if not claims:
        for idx, cl in enumerate(analysis.get("claim_assessments", []), 1):
            cid = str(cl.get("claim_id", f"claim_{idx}"))[:64]
            v = cl.get("verdict", "insufficient_evidence")
            if v not in VERDICTS:
                v = "insufficient_evidence"
            c = cl.get("confidence", 0.5)
            try:
                c = max(0.0, min(1.0, float(c)))
            except (ValueError, TypeError):
                c = 0.5
            claims.append({
                "claim_id": cid,
                "verdict": v,
                "confidence": round(c, 4),
                "evidence_refs": list(dict.fromkeys(all_evidence_refs))[:30],
            })
    claims = claims[:5]

    # data_conflicts
    conflicts = []
    for dc in analysis.get("data_conflicts", []):
        if isinstance(dc, dict):
            sources = dc.get("sources", ["source_a", "source_b"])
            sources = [str(s)[:80] for s in sources][:5]
            if len(sources) < 2:
                sources = ["source_a", "source_b"]
            conflicts.append({
                "field": str(dc.get("field", "unknown"))[:100],
                "sources": sources,
                "selected_source": str(dc.get("selected_source", ""))[:80] if dc.get("selected_source") else None,
                "resolution_code": str(dc.get("resolution_code", "manual_review"))[:80],
            })
    conflicts = conflicts[:5]

    return {
        "schema_version": "day09-l3a-output-v2",
        "case_id": case_id,
        "assessment": {"primary_issue": pi, "case_status": cs, "confidence": round(conf, 4)},
        "affected_entities": affected_entities,
        "claim_assessments": claims,
        "root_cause_analysis": {"ranked_causes": ranked, "responsible_parties": parties},
        "evidence_refs": list(dict.fromkeys(all_evidence_refs))[:30],
        "data_conflicts": conflicts,
        "financial_resolution": {"currency": "BRL", "recommended_refund_brl": round(refund, 2), "refund_lines": refund_lines},
        "resolution_actions": actions,
    }


# ──────────────────────────────────────────────
#  MAIN ENTRY: solve_case
# ──────────────────────────────────────────────

async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Implement the L3A coordinator and specialist-agent workflow."""
    case_id = case["case_id"]
    policy_version = case.get("policy_version", "EC_POLICY_V1")
    customer_req = case.get("customer_request", {})
    raw_claims = customer_req.get("claims", [])
    available_tools = await gateway.list_tools()

    # Phase 1: Coordinator parses complaint
    parsed = await coordinator_parse(case)
    order_ids = parsed.get("order_ids", [])
    claimed_order_id = customer_req.get("claimed_order_id") or case.get("order_id")
    if claimed_order_id and claimed_order_id not in order_ids:
        order_ids.insert(0, claimed_order_id)

    # Phase 2: Dispatch to specialist agents
    trace.emit(case_id=case_id, event_type="task_assigned", actor="coordinator", target="order-agent")
    order_data = await order_agent(case_id, order_ids, gateway, trace, available_tools)

    trace.emit(case_id=case_id, event_type="handoff", actor="order-agent", target="payment-agent")
    trace.emit(case_id=case_id, event_type="task_assigned", actor="coordinator", target="payment-agent")
    payment_data = await payment_agent(case_id, order_ids, gateway, trace, available_tools)

    trace.emit(case_id=case_id, event_type="handoff", actor="payment-agent", target="shipment-agent")
    trace.emit(case_id=case_id, event_type="task_assigned", actor="coordinator", target="shipment-agent")
    shipment_data = await shipment_agent(case_id, order_ids, gateway, trace, available_tools)

    trace.emit(case_id=case_id, event_type="handoff", actor="shipment-agent", target="policy-agent")
    trace.emit(case_id=case_id, event_type="task_assigned", actor="coordinator", target="policy-agent")
    policy_data = await policy_agent(case_id, policy_version, gateway, trace, available_tools)

    # Phase 3: Collect evidence refs
    all_refs: list[str] = []
    for d in [order_data, payment_data, shipment_data, policy_data]:
        all_refs.extend(d.get("evidence_refs", []))
    all_refs = list(dict.fromkeys(all_refs))

    # Phase 4: Build affected_entities
    item_ids, seller_ids, pay_refs, ship_ids = [], [], [], []
    for o in order_data.get("orders", []):
        if isinstance(o, dict) and o.get("seller_id"):
            seller_ids.append(str(o["seller_id"]))
    for it in order_data.get("items", []):
        if isinstance(it, dict):
            for k in ("order_item_id", "item_id", "product_id"):
                if k in it and it[k]:
                    item_ids.append(str(it[k]))
                    break
            if it.get("seller_id"):
                seller_ids.append(str(it["seller_id"]))
    for p in payment_data.get("payments", []):
        if isinstance(p, dict):
            for k in ("payment_sequential", "payment_id", "payment_type"):
                if k in p and p[k]:
                    pay_refs.append(str(p[k]))
                    break
    for s in shipment_data.get("shipments", []):
        if isinstance(s, dict):
            for k in ("shipment_id", "tracking_number", "order_status"):
                if k in s and s[k]:
                    ship_ids.append(str(s[k]))
                    break

    entities = {
        "order_ids": list(dict.fromkeys(order_ids))[:20],
        "item_ids": list(dict.fromkeys(item_ids))[:20],
        "seller_ids": list(dict.fromkeys(seller_ids))[:20],
        "payment_references": list(dict.fromkeys(pay_refs))[:20],
        "shipment_ids": list(dict.fromkeys(ship_ids))[:20],
    }

    # Phase 5: LLM Analysis
    trace.emit(case_id=case_id, event_type="handoff", actor="policy-agent", target="coordinator")
    analysis = await analyze_case(case, parsed, order_data, payment_data,
                                  shipment_data, policy_data, all_refs, raw_claims)

    # Phase 6: Policy Decision
    decision_code = analysis.get("primary_issue", "INSUFFICIENT_EVIDENCE")
    if not isinstance(decision_code, str):
        decision_code = "INSUFFICIENT_EVIDENCE"
    decision_code = re.sub(r"[^A-Za-z0-9_]", "_", decision_code).upper()[:80]
    trace.emit(case_id=case_id, event_type="policy_decided", actor="coordinator",
               decision_code=decision_code)

    # Phase 7: Verification
    trace.emit(case_id=case_id, event_type="handoff", actor="coordinator", target="verifier")
    output = verify_and_fix(analysis, case_id, all_refs, entities, raw_claims)
    trace.emit(case_id=case_id, event_type="verification_completed", actor="verifier")

    return output
