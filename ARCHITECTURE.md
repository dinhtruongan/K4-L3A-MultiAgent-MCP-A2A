# L3A Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 1. System overview

Mỗi case đi qua một chuỗi tuyến tính. Coordinator không gọi tool và không kết
luận; nó chỉ giao việc và chốt. Specialist báo cáo sự kiện, Policy Agent là nơi
duy nhất ra phán quyết, Verifier có quyền phủ quyết trước khi ghi file.

```text
inputs/<case_id>.json
        │
        ▼
   COORDINATOR ── task_assigned ──┬──► ORDER AGENT      get_order, get_order_items
        │                         ├──► PAYMENT AGENT    get_payment_timeline,
        │                         │                     get_refund_timeline
        │                         └──► SHIPMENT AGENT   get_shipment_summary
        │                                   │
        │                    mọi tool call đi qua MCP Evidence Gateway
        │                    → evidence_ref + result_hash + audit phía server
        │                                   │
        │◄────────── handoff (facts + evidence_refs) ───┘
        │
        ├── handoff ──► POLICY AGENT   get_policy → bảng luật EC_POLICY_V1
        │                    │          scope evidence → detect issue → tra policy
        │                    │          policy_decided
        │                    ▼
        │               VERIFIER        9 invariant, không thấy suy luận của Policy
        │                    │          verification_completed
        │◄───── handoff ─────┘
        ▼
outputs/<case_id>.json        traces/trace.jsonl
```

Điểm thiết kế trung tâm: **toàn bộ output phụ thuộc vào đúng một quyết định** —
chọn `primary_issue`. Mọi trường còn lại (`case_status`, `recommended_refund_brl`,
`responsible_parties`, `resolution_actions`) được tra thẳng từ entry tương ứng
trong bảng policy, nên chúng không thể lệch nhau.

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Output/handoff |
| --- | --- | --- | --- |
| Coordinator | case JSON | Giao việc, gom facts, chốt. Không gọi tool, không phán quyết. | `task_assigned`, `case_finalized` |
| Order/item | `case_id`, `order_id` | Xác lập đơn có tồn tại, trạng thái, item, seller, tổng tiền | order row + item rows → coordinator |
| Payment | `case_id`, `order_id` | Xác lập đã thu bao nhiêu, vòng đời hoàn tiền | payment timeline + refund timeline → coordinator |
| Shipment | `case_id`, `order_id` | Xác lập giao hàng có đúng cam kết không, ai làm trễ | shipment summary → coordinator |
| Policy | facts đã scope + `policy_version` | Phân loại issue từ evidence, tra bảng luật, dựng output | output draft → verifier |
| Verifier | output draft + evidence | Kiểm tra 9 invariant chéo trường | PASS, hoặc hạ cấp về `insufficient_evidence` |

Quyền gọi tool được cưỡng chế trong code, không chỉ là quy ước — xem
`TOOL_OWNERSHIP` trong `src/student_agent/agents.py`. Gọi ngoài phạm vi sẽ raise
`EvidenceCollectorError`. Không agent nào được gọi mọi tool.

Ba tool không dùng: `get_sellers` (seller_id đã có trong `get_order_items`),
`get_product_context` và `get_customer_history` (không có tác dụng với bất kỳ
nhánh nào của cây quyết định L3A). Gọi chúng chỉ làm giảm evidence precision.

## 3. A2A protocol

Các agent ở chung process và trao đổi qua shared case state, nhưng ranh giới
message được giữ như message passing: mỗi lần chuyển giao có actor, target và
`decision_code` quan sát được, và được emit thành trace event.

- **Envelope**: `{actor, target, case_id, decision_code}` — chính là hình dạng
  của một trace event `handoff` / `task_assigned`.
- **Correlation**: mọi event và mọi MCP call mang `case_id`. `EvidenceCollector`
  được tạo mới cho từng case và giữ cache riêng, nên evidence ref không thể rò
  sang case khác.
- **Điều kiện handoff**: specialist handoff về coordinator ngay khi facts của
  miền mình đã xác lập; coordinator handoff sang policy khi đủ cả ba miền;
  policy handoff sang verifier khi có draft.
- **Timeout**: kế thừa timeout của gateway (300s tổng, 30s connect/write/pool).
- **Chống vòng lặp**: đồ thị handoff là DAG tuyến tính, không có cạnh quay lui.
  Verifier không trả ngược về Policy để sửa; khi invariant fail nó hạ cấp kết quả
  một lần rồi kết thúc. Số hop mỗi case là hằng số, không phụ thuộc dữ liệu.

## 4. Evidence lifecycle

1. **Validate**: `EvidenceGateway.call` kiểm tra mọi response theo
   `mcp-evidence-response-v1` trước khi trả về. Response sai contract raise ngay.
2. **Store**: ref được gom theo domain trong `EvidenceCollector.refs_by_domain`.
3. **Scope**: mỗi nguồn trả về hai lớp dữ liệu chồng nhau — vòng đời của chính
   đơn này, và các dòng thuộc kịch bản khác. `scope_evidence` chỉ giữ lớp thứ
   nhất, theo các mốc neo sau:

   | Loại dòng | Neo vào | Cửa sổ |
   | --- | --- | --- |
   | item row | `order_purchase_timestamp` | ±7 ngày quanh `shipping_limit_date` |
   | payment event | `order_approved_at` | ±3 ngày |
   | shipment event | `order_delivered_customer_date` (hoặc estimated) | ±3 ngày |
   | refund event | `[purchase, delivery + 14 ngày]` | — |

   Cửa sổ thời gian là điều kiện cần, không phải điều kiện đủ. Một event còn phải
   **không mâu thuẫn với timeline của chính đơn hàng**: event `delivered_late`
   trên một đơn giao trước hạn bị loại dù nó rơi sát ngày giao, vì đơn giao đúng
   hạn thì không thể giao trễ. Timeline của đơn thắng event chống lại nó.

   Dòng bị loại **không bị vứt im lặng**: số lượng được ghi vào `data_conflicts`
   với `resolution_code` là `SCOPED_BY_ORDER_WINDOW_DROPPED_<n>`.
4. **Map**: `RELEVANT_DOMAINS` quyết định domain nào thật sự hỗ trợ từng verdict.
   Chỉ ref thuộc domain liên quan mới được trích vào `evidence_refs`, để giữ
   precision. Mọi ref được trích đều đã xuất hiện trong trace.
5. **Emit**: `tool_result_consumed` được emit bởi chính specialist đã tiêu thụ
   evidence, kèm `evidence_refs`.

Evidence không bao giờ tái sử dụng giữa các case: collector là per-case.

## 5. Failure policy

| Failure | Retry? | Fallback | Trace event/code |
| --- | --- | --- | --- |
| MCP timeout | Không (gateway đã có timeout 300s) | Tool vào `missing_tools`; nếu thuộc `REQUIRED_TOOLS` thì confidence −0.20 | không emit `tool_result_consumed` |
| Not found (`get_refund_timeline` lỗi khi đơn không có refund) | Không — retry vô ích | Coi là "không có refund event", **không** suy ra "chưa hoàn tiền" | không emit; không tính là thiếu bằng chứng |
| Thiếu tool bắt buộc (`get_order`, `get_payment_timeline`, `get_policy`) | Không | Raise `EvidenceCollectorError` — thà dừng còn hơn đoán | — |
| Source conflict (dòng ngoài cửa sổ) | Không | Chọn lớp trong cửa sổ, khai báo trong `data_conflicts` | `policy_decided` kèm `dropped_rows` |
| Invalid specialist result (invariant fail) | Không | Hạ cấp về `insufficient_evidence` / `needs_investigation` / refund 0 | `verification_completed` với `VERIFY_FAIL_<INV>` |

Nguyên tắc bất di dịch: **không chuyển missing evidence thành dữ liệu phỏng
đoán**. Số tiền chỉ đến từ bảng policy, không bao giờ từ suy đoán.

## 6. Verification invariants

Verifier chỉ nhìn output cuối và evidence đã scope, không thấy đường đi của
Policy Agent, nên một lập luận nghe hợp lý nhưng sai không thuyết phục được nó.

| ID | Kiểm tra |
| --- | --- |
| `INV_REFUND_LINES_MISMATCH` | `sum(refund_lines) == recommended_refund_brl` (±0.01) |
| `INV_REFUND_WITHOUT_ACTION_STATUS` | refund > 0 ⟹ `case_status == action_required` |
| `INV_NO_ACTION_WITH_REFUND` | `case_status == no_action` ⟹ refund == 0 |
| `INV_REFUND_EXCEEDS_CAPTURED` | refund ≤ tổng đã thu trong cửa sổ |
| `INV_NO_EVIDENCE` | `evidence_refs` không rỗng |
| `INV_NO_ORDER_ENTITY` | `affected_entities.order_ids` không rỗng |
| `INV_SELLER_NOT_IN_ENTITIES` | `party_type == seller` ⟹ `party_id ∈ seller_ids` |
| `INV_REFUND_LINE_ENTITY_UNKNOWN` | `refund_lines[].entity_id ∈ item_ids` |
| `INV_OVERCONFIDENT_INSUFFICIENT` | `insufficient_evidence` ⟹ confidence ≤ 0.5 |
| `INV_DUPLICATE_ACTIONS` | `resolution_actions` không có phần tử trùng |

Schema compliance được kiểm ba lớp độc lập: `TraceWriter.emit` validate trước khi
ghi từng dòng trace, `cli.py` validate output ngay sau `solve_case`, và
`submission.py` validate lại toàn bộ trước khi đóng gói.

### Quyết định có thể gây tranh luận

Bảng policy giống hệt nhau ở mọi case, nên `party_id` của `party_type == "seller"`
trong đó là giá trị mẫu, không phải seller của case đang xét. Chúng tôi thay bằng
seller thật lấy từ `get_order_items`, vì nêu tên một seller không xuất hiện trong
`affected_entities` sẽ tự mâu thuẫn. Xem `resolve_parties` trong `policy.py`.

`affected_entities.shipment_ids` luôn rỗng: không nguồn evidence nào trả về định
danh shipment. Tự sinh một id sẽ là bịa entity.

## 7. Reproducibility

- **Không dùng LLM ở bất kỳ bước nào.** Không model, không prompt, không nhiệt độ
  sinh, không random seed. Cùng evidence → cùng output, mọi lần chạy.
- Dependency pin theo `pyproject.toml` (`mcp>=2,<3`, `jsonschema[format]>=4.25,<5`,
  `httpx2>=2,<3`, `python-dotenv>=1.1,<2`); Python 3.11.
- Concurrency: tuần tự, một kết nối MCP dùng chung cho cả 100 case. Khoảng 6 tool
  call mỗi case.
- Lệnh chạy:
  ```bash
  day09 validate-inputs
  day09 run
  day09 validate
  day09 package --output dist/submission.zip
  ```
- API key chỉ đọc từ `.env`, không bao giờ vào output, trace hay ZIP;
  `submission.py` quét lại bằng `SECRET_PATTERN` trước khi đóng gói.
