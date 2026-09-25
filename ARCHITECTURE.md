# L3A Architecture Record

## 1. System overview

```text
Input (case JSON)
    → Coordinator Agent (LLM: parse complaint, extract entity IDs)
        → Order Agent (MCP: get_order, get_order_items)
        → Payment Agent (MCP: get_payment)
        → Shipment Agent (MCP: get_shipment)
        → Policy Agent (MCP: get_policy)
    → Coordinator (LLM: analyze & synthesize)
    → Verifier (deterministic schema validation & consistency checks)
    → Output JSON + Trace JSONL
```

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Output/handoff |
| --- | --- | --- | --- |
| Coordinator | Case JSON | Parse complaint, extract IDs, dispatch, aggregate analysis | Parsed info → specialists; Analysis → verifier |
| Order Agent | order_ids | Gọi get_order, get_order_items | Order & item data + evidence_refs |
| Payment Agent | order_ids | Gọi get_payment | Payment data + evidence_refs |
| Shipment Agent | order_ids | Gọi get_shipment | Shipment data + evidence_refs |
| Policy Agent | issue_type | Gọi get_policy | Policy data + evidence_refs |
| Verifier | Analysis result | Schema validation, consistency checks, fix invalid values | Final output dict |

## 3. A2A protocol

Message passing qua function calls trong Python. Correlation bằng `case_id`.
Không có vòng lặp — luồng là one-pass: coordinator → specialists → coordinator → verifier.

## 4. Evidence lifecycle

1. MCP response được validate bởi `Contracts.validate_evidence()`
2. `evidence_ref` được extract và lưu vào list
3. Evidence data được tổng hợp cho LLM analysis
4. `tool_result_consumed` event được emit cho mỗi MCP call
5. Evidence refs được map vào output `evidence_refs` và `claim_assessments`

## 5. Failure policy

| Failure | Retry? | Fallback | Trace event/code |
| --- | --- | --- | --- |
| MCP timeout | No | Skip, continue with partial data | Error logged (ignored silently for completion) |
| Not found | No | Skip entity | Error logged (ignored silently for completion) |
| Source conflict | No | Use LLM to resolve | data_conflicts in output |
| Invalid specialist result | No | Default values via verifier | verification_completed |

## 6. Verification invariants

- Schema compliance (l3a-output-v2)
- primary_issue ∈ allowed enum
- confidence ∈ [0, 1]
- cause_code matches ^[A-Z][A-Z0-9_]{2,79}$
- party_type ∈ allowed enum
- currency = "BRL"
- refund_brl ≥ 0
- All evidence_refs are real (from MCP)
- resolution_actions: max 8, max 80 chars each

## 7. Reproducibility

- Model: Configured via environment variables (LLM_PROVIDER, LLM_MODEL), e.g. Llama 3.1 8B Instruct via Groq API or GPT-4o-mini via OpenAI API.
- Temperature: 0.1
- Concurrency: sequential (1 case at a time)
- Run command: `day09 run`
- Dependencies: see pyproject.toml + openai package
