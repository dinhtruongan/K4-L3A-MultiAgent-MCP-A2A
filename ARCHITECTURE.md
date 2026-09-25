# L3A Architecture Record

Tài liệu mô tả quyết định có thể kiểm chứng của hệ thống; không chứa prompt bí mật hay chain-of-thought.
Toàn bộ agent là Python async state-machine tất định (không dùng LLM), nên cùng input + cùng evidence
luôn cho cùng output.

## 1. System overview

```text
inputs/<case_id>.json
        │  (day09 run → cli._run → emit case_received)
        ▼
  Coordinator ──task_assigned──▶ order-agent ──MCP: get_order, get_order_items?, get_seller──┐
      ▲  ◀──────handoff (FINDINGS_READY)────────┘                                          │
      │ ──task_assigned──▶ payment-agent ──MCP: get_payment, get_refund?─────────────────── ┤
      │  ◀──────handoff────────────┘                                                        │ tool_result_consumed
      │ ──task_assigned──▶ shipment-agent ──MCP: get_shipment───────────────────────────── ┤ (1 event / evidence_ref)
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
| Policy (`policy-agent`) | Toàn bộ facts | Phân loại `primary_issue` bằng bộ luật R10–R99, tra `get_policy`, quyết định trách nhiệm/tiền hoàn/action | `policy_decided`, `handoff VERIFY_DECISION` |
| Verifier (`verifier-agent`) | `Decision` + ledger | Kiểm invariant, sửa/hạ cấp quyết định, hiệu chuẩn confidence | `verification_completed`, `handoff VALIDATED_OUTPUT` |

Quyền gọi tool (least privilege — mỗi specialist chỉ thấy domain của mình, tên tool lấy từ tool discovery):

| Actor | Domain được phép | Tool ứng viên (theo thứ tự ưu tiên) |
| --- | --- | --- |
| order-agent | order, item, seller | `get_order`; `get_order_items`/`get_items`/`get_item`; `get_seller` |
| payment-agent | payment, refund | `get_payment`/`get_payments`; `get_refund`/`get_refunds` |
| shipment-agent | shipment | `get_shipment`/`get_tracking`/`get_delivery` |
| policy-agent | policy | `get_policy`/`get_refund_policy` |
| coordinator, verifier | — | không gọi MCP |

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
4. `CaseState.ledger` chỉ tồn tại trong một lần `solve_case` → không thể tái sử dụng giữa các case.
5. Khi dựng output, chỉ trích dẫn evidence thuộc domain hỗ trợ kết luận (`ISSUE_DOMAINS` trong
   `policy.py`, ví dụ `late_delivery_seller` → order/item/shipment/seller/policy; lỗi thanh toán
   không trích shipment). Coordinator kiểm `output.evidence_refs ⊆ ledger` trước khi trả về.

## 5. Failure policy

| Failure | Retry? | Fallback | Trace event/code |
| --- | --- | --- | --- |
| MCP timeout / lỗi transport | Có, tối đa 3 lần, backoff 0.5 s → 1.5 s (call đọc, idempotent) | Bỏ domain đó; verifier có thể hạ cấp thành `insufficient_evidence` | handoff `FINDINGS_EMPTY`; lỗi `tool_error:*` → trừ confidence |
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
7. **Confidence** ∈ [0.05, 0.95]: base theo issue (0.92 cho canceled-paid … 0.35 cho insufficient),
   trừ cho warning của evidence (tối đa 0.15), mỗi conflict 0.08, thiếu timestamp bàn giao 0.15,
   lỗi tool (tối đa 0.1), claim khách không khớp issue 0.07. Không bao giờ 1.0.

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
