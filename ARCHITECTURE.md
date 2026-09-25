# L3A Architecture Record

Tài liệu mô tả quyết định có thể kiểm chứng của hệ thống; không chứa prompt bí mật hay chain-of-thought.
Toàn bộ agent là Python async state-machine tất định (không dùng LLM), nên cùng input + cùng evidence
luôn cho cùng output.

## 1. System overview

```text
inputs/<case_id>.json
        │  (day09 run → cli._run → emit case_received)
        ▼
  Coordinator ──task_assigned──▶ order-agent ──MCP: get_order, get_order_items, get_sellers─┐
      ▲  ◀──────handoff (FINDINGS_READY)────────┘                                          │
      │ ──task_assigned──▶ payment-agent ──MCP: get_payment_timeline, get_refund_timeline── ┤
      │  ◀──────handoff────────────┘                                                        │ tool_result_consumed
      │ ──task_assigned──▶ shipment-agent ──MCP: get_shipment_summary───────────────────── ┤ (1 event / evidence_ref)
      │  ◀──────handoff────────────┘                                                        │
      │ ──handoff (DECIDE_POLICY)──▶ policy-agent ──MCP: get_policy──────────────────────── ┘
      │                               │ policy_decided (decision_code = primary_issue)
      │                               ▼ handoff (VERIFY_DECISION)
      │                          verifier-agent ── verification_completed (PASSED | REVISED)
      └──────── handoff (VALIDATED_OUTPUT) ◀──┘
        │  schema check → outputs/<case_id>.json → emit case_finalized
        ▼
  traces/trace.jsonl, outputs/*.json → day09 validate → day09 package
```

Mã nguồn:

| File | Vai trò |
| --- | --- |
| `src/student_agent/workflow.py` | `solve_case`, `Coordinator`, `PolicyAgent`, dựng output |
| `src/student_agent/specialists.py` | `OrderItemAgent`, `PaymentAgent`, `ShipmentAgent`, retry, map tham số tool |
| `src/student_agent/policy.py` | Bộ luật phân loại issue, trách nhiệm, tiền hoàn, action |
| `src/student_agent/verifier.py` | Invariant chéo trường + hiệu chuẩn confidence |
| `src/student_agent/state.py` | `CaseState`, `AgentMessage` (A2A envelope), `Evidence`, `Finding`, `Decision` |
| `src/student_agent/evidence.py` | Đọc payload MCP phòng thủ (alias field, parse ngày/tiền) |
| `src/student_agent/mcp_gateway.py` | MCP client, tool discovery kèm input schema, phân loại lỗi |

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Output/handoff |
| --- | --- | --- | --- |
| Coordinator | Case input (`case_id`, message, ID gợi ý) | Trích ID + loại claim (chỉ là gợi ý), điều phối tuần tự, truyền context (order_id, payment_ref, shipment_id), kiểm hop limit, dựng output | `task_assigned` tới specialist; `handoff` tới policy; nhận `VALIDATED_OUTPUT` |
| Order/item (`order-agent`) | `order_id` | Trạng thái đơn, timeline, item (price, freight, shipping_limit), seller_ids, order_total | `Finding` + `handoff FINDINGS_READY/EMPTY` |
| Payment (`payment-agent`) | `order_id` / payment ref | Sổ thanh toán, paid_total, số lần/loại thanh toán, refund và trạng thái | như trên |
| Shipment (`shipment-agent`) | `order_id` / shipment_id | Thời điểm giao carrier, giao khách, ETA, carrier, lý do trễ | như trên |
| Policy (`policy-agent`) | Toàn bộ facts (đã lọc phạm vi thời gian) | Phân loại `primary_issue` bằng bộ luật R10–R50, tra `get_policy` và áp `rules[primary_issue]` (status, action, refund, bên chịu trách nhiệm) | `policy_decided`, `handoff VERIFY_DECISION` |
| Verifier (`verifier-agent`) | `Decision` + ledger | Kiểm invariant, sửa/hạ cấp quyết định, hiệu chuẩn confidence | `verification_completed`, `handoff VALIDATED_OUTPUT` |

Quyền gọi tool (least privilege — mỗi specialist chỉ thấy domain của mình, tên tool lấy từ tool discovery):

| Actor | Domain được phép | Tool ứng viên (theo thứ tự ưu tiên) |
| --- | --- | --- |
| order-agent | order, item, seller | `get_order`; `get_order_items`; `get_sellers` |
| payment-agent | payment, refund | `get_payment_timeline` (fallback `get_order_payments`); `get_refund_timeline` |
| shipment-agent | shipment | `get_shipment_summary` |
| policy-agent | policy | `get_policy(policy_version)` |
| coordinator, verifier | — | không gọi MCP |

Không dùng `get_product_context`, `get_customer_history` (không đóng góp vào kết luận; tool sau
cần `customer_unique_id` không có trong input/evidence). Mỗi case: 7 call.

Tham số tool được điền từ `inputSchema` của tool discovery: chỉ dùng giá trị có trong case input hoặc
trong evidence trước đó; nếu thiếu tham số bắt buộc thì bỏ qua call (ghi `missing_argument:*`),
không đoán.

## 3. A2A protocol

- Envelope `AgentMessage{case_id, sender, recipient, intent, hop, payload, message_id}`; `case_id` là
  correlation key, `message_id` được ghi vào `attributes` của trace.
- Mỗi lần gửi message → 1 trace event: coordinator→specialist là `task_assigned`, các chuyển giao
  còn lại là `handoff` (`actor`=sender, `target`=recipient, `decision_code`=intent,
  `evidence_refs` = evidence được bàn giao).
- Điều kiện handoff: specialist luôn trả về (kể cả khi rỗng: `FINDINGS_EMPTY`) để coordinator không
  bị treo; policy chỉ chạy sau khi đủ 3 specialist; verifier chỉ chạy khi có `Decision`.
- Tránh vòng lặp: luồng là DAG cố định, verifier sửa trực tiếp (không gửi ngược policy), và
  `MAX_HOPS = 12` (luồng chuẩn dùng 9 hop) — vượt thì raise.
- Timeout: httpx timeout 300 s/request, connect 30 s (trong `connect_gateway`).
- Trace chỉ chứa sự kiện quan sát được và decision code; không ghi nội dung suy luận.

## 4. Evidence lifecycle

1. `EvidenceGateway.call` luôn gửi `case_id` của case hiện tại và validate envelope theo
   `mcp-evidence-response-v1.schema.json` (sai schema → coi như lỗi, không dùng).
2. Specialist tạo `Evidence{ref, tool, domain, data, actor, warnings}` nguyên vẹn từ envelope —
   `evidence_ref` không bao giờ được sinh/sửa.
3. Ngay khi dùng evidence, specialist emit `tool_result_consumed` với `tool_name` và
   `evidence_refs=[ref]`.
4. **Temporal scope**: payload có thể chứa dòng không thuộc kịch bản đang điều tra (item,
   payment event, refund event, shipping limit, shipment event có timestamp nằm ngoài vòng đời
   đơn). Mỗi specialist chỉ dùng dòng có timestamp trong `[order_purchase_timestamp − 1h,
   opened_at]`; dòng không có timestamp được giữ. Số dòng bị loại ghi vào attribute
   `out_of_scope_rows` của `verification_completed`.
5. `CaseState.ledger` chỉ tồn tại trong một lần `solve_case` → không thể tái sử dụng giữa các case.
6. Khi dựng output, chỉ trích dẫn evidence thuộc domain hỗ trợ kết luận (`ISSUE_DOMAINS` trong
   `policy.py`, ví dụ `late_delivery_seller` → order/item/shipment/seller/policy; lỗi thanh toán
   không trích shipment). Coordinator kiểm `output.evidence_refs ⊆ ledger` trước khi trả về.

### Bộ luật phân loại (`policy.classify`, theo thứ tự)

| Rule | Điều kiện (chỉ trên dữ liệu trong phạm vi) | Issue |
| --- | --- | --- |
| R10/R11 | refund event `failed` / `pending` | `refund_failed` / `refund_pending` |
| R21/R22 | `order_status` = canceled / unavailable và đã capture tiền | `canceled_order_paid` / `unavailable_order_paid` |
| R30 | payment lifecycle có `reconciliation_mismatch` đang mở | `payment_mismatch` |
| R40/R41 | giao khách sau ETA; handover carrier > shipping limit → seller, ngược lại → logistics | `late_delivery_seller` / `late_delivery_logistics` |
| R31/R32 | ≥2 capture: tổng một nhóm = tổng đơn → split hợp lệ; hai capture bằng nhau nhưng vượt tổng → trùng | `valid_split_payment` / `duplicate_charge` |
| R50 | không có bất thường | `unsupported_claim` |

Sau đó `resolve` áp `get_policy.rules[issue]`: `case_status`, `recommended_action`
(→ `resolution_actions`), `refund_brl` (→ refund line cho order, bị verifier cap ở số đã trả),
`responsible_parties` (party `seller` luôn lấy `seller_id` từ evidence của case, không lấy ID mẫu
trong policy). Thiếu policy → bảng mặc định + tính tiền từ evidence, trừ confidence.

## 5. Failure policy

| Failure | Retry? | Fallback | Trace event/code |
| --- | --- | --- | --- |
| MCP timeout / lỗi transport | Có, tối đa 3 lần, backoff 0.5 s → 1.5 s (call đọc, idempotent) | Bỏ domain đó; verifier có thể hạ cấp thành `insufficient_evidence` | handoff `FINDINGS_EMPTY`; lỗi `tool_error:*` → trừ confidence |
| Tool trả lỗi (vd. `get_refund_timeline` khi đơn không có refund) | Không (lỗi tất định từ gateway) | Coi như không có bản ghi | `no_records:*`, không trừ confidence |
| Not found | Không | Không dùng domain đó; không thay bằng dữ liệu phỏng đoán | `not_found:*`; `verification_completed` `REVISED` nếu thiếu evidence bắt buộc |
| 403 / sai scope | Không | Như not found | `forbidden:*` |
| Envelope sai schema | Không | Loại evidence | `invalid_envelope:*` |
| Source conflict (order vs shipment lệch > 24 h; số tiền khách nói vs sổ thanh toán) | — | Ưu tiên shipment tracking / payment ledger | `data_conflicts[]` + trừ 0.08 confidence mỗi conflict |
| Invalid specialist result (thiếu domain bắt buộc cho issue) | — | Verifier hạ cấp `insufficient_evidence`, `needs_investigation` | `verification_completed` `decision_code=REVISED` |

## 6. Verification invariants

Kiểm trước khi finalize (`verifier.py` + `Coordinator.solve`):

1. **Evidence bắt buộc**: mỗi issue có domain bắt buộc (`REQUIRED_DOMAINS`), thiếu → hạ cấp.
2. **Money**: refund line ≥ 0, làm tròn 2 chữ số, tổng ≤ `paid_total`;
   `recommended_refund_brl = Σ refund_lines`.
3. **Status/refund/action**: `no_action` ⇒ refund = 0; refund > 0 ⇒ `action_required` và có action
   `refund_*`; không có refund ⇒ không có action `refund_*`; `no_action` không có action escalate.
4. **Responsibility**: `seller` chỉ dùng `seller_id` có trong evidence; lỗi người bán ⇒ không gán
   `logistics_provider`; actions không trùng lặp.
5. **Schema**: output validate theo `l3a-output-v2.schema.json` trong workflow và lần nữa ở CLI.
6. **Entity scope / ownership**: `evidence_refs ⊆ ledger` của chính case.
7. **Confidence** ∈ [0.05, 0.95]: 0.93 khi evidence đầy đủ và khớp claim; 0.3 cho
   `insufficient_evidence`; trừ cho warning (tối đa 0.15), mỗi conflict 0.05, thiếu timestamp
   bàn giao / actor sự kiện trễ mâu thuẫn 0.15, thiếu rule policy 0.1, lỗi transport (tối đa 0.1),
   issue khác claim topic 0.2. Không bao giờ 1.0.
8. **Claim assessments**: claim topic = issue → `supported`, khác → `unsupported`;
   `requested_full_refund` → `supported` (hoàn toàn bộ: canceled/unavailable/refund_*),
   `partially_supported` (late/duplicate/mismatch), `unsupported` (split/unsupported).

## 7. Reproducibility

- Không dùng LLM, không random: kết quả chỉ phụ thuộc input + evidence MCP.
- Python 3.11; dependency theo `pyproject.toml` (`mcp>=2,<3`, `httpx2`, `jsonschema`).
- Concurrency: tuần tự từng case, tuần tự từng agent (1 MCP session).
- Lệnh chạy:

  ```bash
  python -m pip install -e ".[dev]"
  day09 validate-inputs && day09 mcp-tools
  day09 run && day09 validate
  day09 package --output dist/submission.zip
  ```

- Test offline (gateway giả, không cần mạng): `pytest -q`.
- API key chỉ nằm trong `.env` (gitignored), không ghi vào output/trace.
