# 📘 Hướng Dẫn Chi Tiết Hoàn Thành Bài Lab Day09 — Multi-Agent MCP + A2A

> **Yêu cầu đặc biệt**: Giảng viên yêu cầu **chỉ sử dụng model nhỏ hơn 10B tham số**.

---

## Mục lục

1. [Tổng quan bài lab](#1-tổng-quan-bài-lab)
2. [Yêu cầu hệ thống & cài đặt môi trường](#2-yêu-cầu-hệ-thống--cài-đặt-môi-trường)
3. [Lựa chọn model nhỏ (< 10B tham số)](#3-lựa-chọn-model-nhỏ--10b-tham-số)
4. [Cấu trúc dự án & hiểu source code](#4-cấu-trúc-dự-án--hiểu-source-code)
5. [Đăng ký team & cấu hình `.env`](#5-đăng-ký-team--cấu-hình-env)
6. [Tải và validate input](#6-tải-và-validate-input)
7. [Hiểu MCP Evidence Gateway](#7-hiểu-mcp-evidence-gateway)
8. [Hiểu Output Schema (JSON)](#8-hiểu-output-schema-json)
9. [Hiểu Trace Event Schema](#9-hiểu-trace-event-schema)
10. [Thiết kế kiến trúc Multi-Agent](#10-thiết-kế-kiến-trúc-multi-agent)
11. [Triển khai `workflow.py` — Code mẫu đầy đủ](#11-triển-khai-workflowpy--code-mẫu-đầy-đủ)
12. [Cập nhật `ARCHITECTURE.md`](#12-cập-nhật-architecturemd)
13. [Chạy, validate & debug](#13-chạy-validate--debug)
14. [Đóng gói & nộp bài](#14-đóng-gói--nộp-bài)
15. [Tiêu chí chấm điểm & chiến lược tối ưu](#15-tiêu-chí-chấm-điểm--chiến-lược-tối-ưu)
16. [Lưu ý quan trọng & sai lầm phổ biến](#16-lưu-ý-quan-trọng--sai-lầm-phổ-biến)

---

## 1. Tổng quan bài lab

### Mục tiêu
Xây dựng hệ thống **multi-agent** điều tra khiếu nại thương mại điện tử (e-commerce complaints), sử dụng dữ liệu từ [Brazilian E-Commerce (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

### Yêu cầu chính
- Đọc yêu cầu (complaint) của khách hàng từ file input
- Lấy dữ liệu có thẩm quyền qua **MCP Evidence Gateway** (không tự bịa data)
- Phối hợp giữa **nhiều agent chuyên biệt** (multi-agent) để phân tích
- Tạo **output JSON** và **trace JSONL** đúng public contract
- **Không được tự đoán dữ liệu** hoặc tạo `evidence_ref` giả

### Luồng xử lý tổng thể

```
Input (case JSON) → Coordinator Agent
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   Order Agent     Payment Agent   Shipment Agent
   (get_order)     (get_payment)   (get_shipment)
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                  Policy Agent (get_policy)
                         │
                         ▼
                  Verifier Agent
                         │
                         ▼
              Output JSON + Trace JSONL
```

---

## 2. Yêu cầu hệ thống & cài đặt môi trường

### 2.1. Yêu cầu
- **Python 3.11** trở lên
- **Git** đã cài sẵn
- **Kết nối internet** (để truy cập MCP Gateway & LLM API)

### 2.2. Tạo virtual environment

**Trên Windows (PowerShell):**
```powershell
cd D:\Repo\K4-L3A-MultiAgent-MCP-A2A-day09

# Tạo venv
python3.11 -m venv .venv

# Kích hoạt (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Nếu bị lỗi Execution Policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

**Trên macOS/Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2.3. Cài đặt dependencies

```bash
# Cài package ở chế độ develop
python -m pip install -e ".[dev]"

# Kiểm tra cài đặt thành công
pytest -q
day09 --help
```

**Output mong đợi khi chạy `day09 --help`:**
```
usage: day09 [-h] [--root ROOT] {validate-inputs,mcp-tools,run,validate,package} ...

Day09 L3A student workflow

positional arguments:
  {validate-inputs,mcp-tools,run,validate,package}
    validate-inputs     validate case-set.json and all 100 inputs
    mcp-tools           authenticate and list discovered MCP tools
    run                 run the implemented workflow for all cases
    validate            validate outputs and observable trace
    package             validate and build the submission ZIP
```

### 2.4. Cài thêm thư viện cho LLM

Tùy thuộc vào model bạn chọn:

```bash
# Nếu dùng OpenAI-compatible API (Groq, Together AI, OpenRouter...)
pip install openai

# Nếu dùng Ollama (chạy local)
pip install ollama
# hoặc dùng thông qua openai client với base_url tùy chỉnh
```

---

## 3. Lựa chọn model nhỏ (< 10B tham số)

> ⚠️ **Giảng viên yêu cầu chỉ dùng model nhỏ hơn 10B tham số.**

### 3.1. Model đề xuất (sắp xếp theo khuyến nghị)

| Model | Tham số | Provider | Ưu điểm | Nhược điểm |
|---|---:|---|---|---|
| **Qwen 2.5 7B Instruct** | 7B | Ollama / Together AI / Groq | Mạnh reasoning, hỗ trợ JSON tốt | Cần GPU 8GB+ nếu chạy local |
| **Llama 3.1 8B Instruct** | 8B | Ollama / Groq / Together AI | Nhanh, chất lượng cao | - |
| **Gemma 2 9B Instruct** | 9B | Ollama / Groq | Tốt cho phân tích | RAM cao hơn |
| **Mistral 7B Instruct v0.3** | 7B | Ollama / Together AI | Nhẹ, nhanh | Kém hơn Qwen/Llama ở structured output |
| **Phi-3.5 Mini Instruct** | 3.8B | Ollama / Azure | Rất nhẹ, vẫn reasoning được | Output JSON đôi khi không ổn định |

### 3.2. Cách sử dụng qua các provider

#### A) Dùng Groq (miễn phí, nhanh — **khuyến nghị**)

1. Đăng ký tại [console.groq.com](https://console.groq.com)
2. Lấy API key
3. Cấu hình `.env`:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxx
```

4. Sử dụng thông qua OpenAI-compatible client:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

response = await client.chat.completions.create(
    model="llama-3.1-8b-instant",  # < 10B ✓
    messages=[...],
    temperature=0.1,
    response_format={"type": "json_object"},
)
```

#### B) Dùng Ollama (chạy local, không cần API key)

1. Cài Ollama: [ollama.com/download](https://ollama.com/download)
2. Pull model:

```bash
ollama pull qwen2.5:7b-instruct
# hoặc
ollama pull llama3.1:8b-instruct-q4_K_M
```

3. Ollama tự host OpenAI-compatible API tại `http://localhost:11434/v1`

```python
client = AsyncOpenAI(
    api_key="ollama",  # Ollama không cần key thật
    base_url="http://localhost:11434/v1",
)

response = await client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[...],
    temperature=0.1,
    response_format={"type": "json_object"},
)
```

#### C) Dùng Together AI / OpenRouter

```dotenv
LLM_PROVIDER=together
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo
TOGETHER_API_KEY=xxx
```

```python
client = AsyncOpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)
```

---

## 4. Cấu trúc dự án & hiểu source code

### 4.1. Cây thư mục

```
K4-L3A-MultiAgent-MCP-A2A-day09/
├── .env                    # Cấu hình bí mật (API keys) — KHÔNG commit
├── .env.example            # Mẫu cấu hình
├── ARCHITECTURE.md         # Mô tả thiết kế (cần cập nhật)
├── README.md               # Hướng dẫn chính thức
├── pyproject.toml          # Cấu hình project Python
├── contracts/              # Public contracts (KHÔNG sửa)
│   ├── schemas/
│   │   ├── l3a-output-v2.schema.json         # ★ Schema output bắt buộc
│   │   ├── mcp-evidence-response-v1.schema.json
│   │   ├── trace-event-v1.schema.json        # ★ Schema trace event
│   │   └── submission-manifest-v2.schema.json
│   ├── registry/variants.json
│   └── scoring/scoring-policy-v2.json        # ★ Trọng số chấm điểm
├── inputs/                 # Dữ liệu input (100 cases, sau khi tải)
│   └── L3A_CASE_001.json ... L3A_CASE_100.json
├── outputs/                # Output của workflow (tự sinh)
├── traces/                 # Trace JSONL (tự sinh)
├── src/student_agent/      # ★ CODE CẦN IMPLEMENT
│   ├── __init__.py         # VARIANT_ID = "l3a"
│   ├── cli.py              # CLI commands (day09 run, validate...)
│   ├── config.py           # Đọc .env settings
│   ├── contracts.py        # Validate output/trace/evidence theo schema
│   ├── cases.py            # Load case-set.json & inputs
│   ├── mcp_gateway.py      # ★ Client MCP (EvidenceGateway)
│   ├── trace.py            # ★ TraceWriter — ghi trace events
│   ├── submission.py       # Đóng gói submission ZIP
│   └── workflow.py         # ★★★ FILE CHÍNH CẦN IMPLEMENT
└── tests/
    ├── test_starter.py
    └── test_release_safety.py
```

### 4.2. Các file quan trọng cần hiểu

#### `workflow.py` — File chính cần code
```python
async def solve_case(case, gateway, trace) -> dict:
    """
    Nhận vào:
      - case: dict chứa thông tin khiếu nại (case_id, customer_message, ...)
      - gateway: EvidenceGateway — gọi MCP tools lấy evidence
      - trace: TraceWriter — ghi trace events cho workflow

    Trả về:
      - dict đúng l3a-output-v2 schema
    """
```

#### `mcp_gateway.py` — Cách gọi MCP tools
```python
# Liệt kê tool
tools = await gateway.list_tools()  # → ["get_order", "get_payment", ...]

# Gọi tool lấy evidence
evidence = await gateway.call(
    "get_order",               # tên tool
    case_id=case["case_id"],   # bắt buộc
    order_id="abc123",         # tham số của tool
)
# evidence = {
#   "schema_version": "day09-mcp-evidence-v1",
#   "evidence_ref": "ev_xxxxx",     # ← Dùng ref này trong output
#   "result_hash": "sha256:...",
#   "domain": "order",
#   "data": { ... },                # ← Dữ liệu thật
#   "warnings": [...]               # (optional)
# }
```

#### `trace.py` — Cách ghi trace
```python
trace.emit(
    case_id="L3A_CASE_001",
    event_type="task_assigned",       # Một trong 7 loại event
    actor="coordinator",              # Tên agent
    target="order-agent",             # (optional) Agent nhận task
    tool_name="get_order",            # (optional) Tên tool MCP
    evidence_refs=["ev_xxx"],         # (optional) Evidence refs
    decision_code="LATE_DELIVERY",    # (optional) Quyết định
)
```

---

## 5. Đăng ký team & cấu hình `.env`

### 5.1. Đăng ký team
1. Mở `/register` trên Competition Workspace
2. Điền tên team, mã học viên, thành viên
3. Nhập registration code lớp
4. Lưu Team API Key `sk-team-...`

### 5.2. Cấu hình `.env`

Sao chép từ template:
```bash
cp .env.example .env
```

Sửa file `.env`:
```dotenv
# Competition (giữ nguyên)
COMPETITION_API_URL=https://n7-competition.pages.dev
COMPETITION_TEAM_API_KEY=sk-team-YOUR_REAL_KEY_HERE
MCP_ENDPOINT=https://day09-competition.34-142-201-239.sslip.io/mcp

# LLM Configuration (chọn một trong các option dưới)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=gsk_your_groq_api_key_here

# Hoặc nếu dùng Ollama local:
# LLM_PROVIDER=ollama
# LLM_MODEL=qwen2.5:7b-instruct
# OLLAMA_BASE_URL=http://localhost:11434/v1
```

> ⚠️ **KHÔNG BAO GIỜ commit file `.env` lên Git!** File này đã có trong `.gitignore`.

---

## 6. Tải và validate input

### 6.1. Tải input
Tải ZIP input **L3A** từ GitHub Release hoặc link giảng viên cung cấp, giải nén vào root repo:

```bash
# Giải nén (thay <version> bằng version thực tế)
unzip l3a-inputs-<version>.zip -d .

# Trên Windows có thể dùng:
# Expand-Archive -Path l3a-inputs-<version>.zip -DestinationPath .
```

### 6.2. Kiểm tra cấu trúc

Sau khi giải nén, cấu trúc phải là:
```
case-set.json          ← File manifest (100 case IDs)
inputs/
├── L3A_CASE_001.json
├── L3A_CASE_002.json
├── ...
└── L3A_CASE_100.json
```

### 6.3. Validate

```bash
day09 validate-inputs
```

Output thành công:
```
OK: l3a / <version> / 100 cases
```

### 6.4. Xem thử 1 case input

```bash
python -c "import json; print(json.dumps(json.load(open('inputs/L3A_CASE_001.json')), indent=2, ensure_ascii=False))"
```

Mỗi case sẽ chứa ít nhất:
- `case_id`: ID duy nhất (ví dụ `"L3A_CASE_001"`)
- `customer_message`: Nội dung khiếu nại bằng tiếng Bồ Đào Nha hoặc tiếng Anh
- Có thể chứa thêm: `order_id`, `customer_id`, hints...

---

## 7. Hiểu MCP Evidence Gateway

### 7.1. MCP là gì?
MCP (Model Context Protocol) cung cấp các **tool** để truy vấn dữ liệu có thẩm quyền. Mọi lời gọi đều được **audit** (ghi log bởi server).

### 7.2. Khám phá tool

```bash
day09 mcp-tools
```

Sẽ hiện các tool có sẵn, ví dụ:
- `get_order` — Lấy thông tin đơn hàng
- `get_order_items` — Lấy danh sách sản phẩm trong đơn
- `get_payment` — Lấy thông tin thanh toán
- `get_shipment` — Lấy thông tin vận chuyển
- `get_seller` — Lấy thông tin người bán
- `get_policy` — Lấy chính sách hoàn tiền/đổi trả
- ...

### 7.3. Domain của evidence

Mỗi evidence response thuộc 1 domain:
```
"order" | "item" | "payment" | "shipment" | "seller" |
"customer" | "product" | "refund" | "policy"
```

### 7.4. Quy tắc BẮT BUỘC khi dùng MCP

| # | Quy tắc | Vi phạm → Hậu quả |
|---|---------|-------------------|
| 1 | Luôn truyền đúng `case_id` | 0 điểm cho case |
| 2 | Dùng tool discovery, KHÔNG đoán tên tool | Tool sai → error |
| 3 | KHÔNG sửa hoặc tự tạo `evidence_ref` | 0 điểm provenance |
| 4 | KHÔNG dùng evidence chéo case | 0 điểm provenance |
| 5 | Chỉ trích dẫn evidence thật sự hỗ trợ kết luận | Giảm evidence score |

---

## 8. Hiểu Output Schema (JSON)

Mỗi case phải tạo 1 file `outputs/<case_id>.json` đúng schema `l3a-output-v2`.

### 8.1. Cấu trúc output bắt buộc

```json
{
  "schema_version": "day09-l3a-output-v2",
  "case_id": "L3A_CASE_001",

  "assessment": {
    "primary_issue": "<primary_issue_enum>",
    "case_status": "action_required | no_action | needs_investigation",
    "confidence": 0.85
  },

  "affected_entities": {
    "order_ids": ["order_abc123"],
    "item_ids": ["item_xyz"],
    "seller_ids": ["seller_001"],
    "payment_references": ["pay_ref_001"],
    "shipment_ids": ["ship_001"]
  },

  "claim_assessments": [
    {
      "claim_id": "claim_late_delivery",
      "verdict": "supported | unsupported | partially_supported | insufficient_evidence",
      "confidence": 0.9,
      "evidence_refs": ["ev_xxxxx"]
    }
  ],

  "root_cause_analysis": {
    "ranked_causes": [
      {"cause_code": "LATE_DELIVERY_LOGISTICS", "rank": 1},
      {"cause_code": "SELLER_DELAY", "rank": 2}
    ],
    "responsible_parties": [
      {"party_type": "logistics_provider", "party_id": null},
      {"party_type": "seller", "party_id": "seller_001"}
    ]
  },

  "evidence_refs": ["ev_xxxxx", "ev_yyyyy"],

  "data_conflicts": [],

  "financial_resolution": {
    "currency": "BRL",
    "recommended_refund_brl": 150.50,
    "refund_lines": [
      {
        "reason_code": "late_delivery_refund",
        "amount_brl": 150.50,
        "entity_id": "order_abc123"
      }
    ]
  },

  "resolution_actions": [
    "Issue full refund to customer",
    "Flag logistics provider for SLA violation"
  ]
}
```

### 8.2. Các giá trị `primary_issue` hợp lệ

| Giá trị | Ý nghĩa |
|---------|---------|
| `canceled_order_paid` | Đơn bị hủy nhưng đã thanh toán |
| `unavailable_order_paid` | Sản phẩm hết hàng nhưng đã thanh toán |
| `late_delivery_seller` | Giao hàng trễ do lỗi người bán |
| `late_delivery_logistics` | Giao hàng trễ do lỗi vận chuyển |
| `valid_split_payment` | Thanh toán chia nhiều lần hợp lệ |
| `payment_mismatch` | Số tiền thanh toán không khớp |
| `duplicate_charge` | Tính phí trùng |
| `refund_pending` | Hoàn tiền đang chờ |
| `refund_failed` | Hoàn tiền thất bại |
| `unsupported_claim` | Khiếu nại không có cơ sở |
| `insufficient_evidence` | Thiếu bằng chứng để kết luận |

### 8.3. `cause_code` pattern

Phải match regex: `^[A-Z][A-Z0-9_]{2,79}$`

Ví dụ hợp lệ: `LATE_DELIVERY`, `PAYMENT_MISMATCH`, `SELLER_DELAY`, `LOGISTICS_FAILURE`

---

## 9. Hiểu Trace Event Schema

Trace ghi lại quy trình làm việc của multi-agent. **Rất quan trọng cho điểm workflow (5%).**

### 9.1. Các loại event bắt buộc

Theo `scoring-policy-v2.json`, cần có đầy đủ các event sau cho MỖI case:

| Event Type | Mô tả | Actor |
|-----------|-------|-------|
| `case_received` | Case được nhận | `coordinator` |
| `task_assigned` | Giao task cho specialist | `coordinator` |
| `tool_result_consumed` | Agent sử dụng evidence từ MCP | specialist agents |
| `handoff` | Chuyển giao giữa agents | agent nguồn → agent đích |
| `policy_decided` | Quyết định chính sách | `policy-agent` hoặc `coordinator` |
| `verification_completed` | Xác minh hoàn thành | `verifier` |
| `case_finalized` | Case kết thúc | `coordinator` |

### 9.2. Ví dụ trace flow cho 1 case

```python
# 1. Case received (đã có sẵn trong cli.py)
trace.emit(case_id=cid, event_type="case_received", actor="coordinator")

# 2. Giao task cho order-agent
trace.emit(case_id=cid, event_type="task_assigned", actor="coordinator", target="order-agent")

# 3. Order-agent dùng evidence
trace.emit(case_id=cid, event_type="tool_result_consumed", actor="order-agent",
           tool_name="get_order", evidence_refs=["ev_xxx"])

# 4. Handoff từ order-agent sang payment-agent
trace.emit(case_id=cid, event_type="handoff", actor="order-agent", target="payment-agent")

# 5. Payment-agent dùng evidence
trace.emit(case_id=cid, event_type="tool_result_consumed", actor="payment-agent",
           tool_name="get_payment", evidence_refs=["ev_yyy"])

# 6. Handoff sang verifier
trace.emit(case_id=cid, event_type="handoff", actor="payment-agent", target="verifier")

# 7. Policy decided
trace.emit(case_id=cid, event_type="policy_decided", actor="coordinator",
           decision_code="REFUND_APPROVED")

# 8. Verification completed
trace.emit(case_id=cid, event_type="verification_completed", actor="verifier")

# 9. Case finalized (đã có sẵn trong cli.py)
trace.emit(case_id=cid, event_type="case_finalized", actor="coordinator")
```

---

## 10. Thiết kế kiến trúc Multi-Agent

### 10.1. Sơ đồ kiến trúc đề xuất

```
┌─────────────────────────────────────────────────────┐
│                   COORDINATOR                       │
│  • Parse customer message (LLM)                     │
│  • Extract entity IDs (order_id, etc.)              │
│  • Dispatch tasks to specialist agents              │
│  • Aggregate results                                │
│  • Final output assembly                            │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
  │ Order   ││ Payment ││Shipment ││ Policy  │
  │ Agent   ││ Agent   ││ Agent   ││ Agent   │
  │         ││         ││         ││         │
  │get_order││get_pay  ││get_ship ││get_pol  │
  │get_items││         ││         ││         │
  └────┬────┘└────┬────┘└────┬────┘└────┬────┘
       │          │          │          │
       └──────────┴──────────┴──────────┘
                         │
                         ▼
               ┌──────────────────┐
               │    VERIFIER      │
               │ • Cross-check    │
               │ • Schema valid   │
               │ • Consistency    │
               └──────────────────┘
```

### 10.2. Vai trò từng Agent

| Agent | MCP Tools | Trách nhiệm |
|-------|-----------|-------------|
| **Coordinator** | Không trực tiếp | Parse message, extract IDs, dispatch, aggregate |
| **Order Agent** | `get_order`, `get_order_items` | Lấy thông tin đơn hàng, items, trạng thái |
| **Payment Agent** | `get_payment` | Lấy thông tin thanh toán, so sánh số tiền |
| **Shipment Agent** | `get_shipment` | Lấy thông tin vận chuyển, ngày giao |
| **Policy Agent** | `get_policy` | Lấy chính sách hoàn tiền/đổi trả |
| **Verifier** | Không | Kiểm tra tính nhất quán, cross-validate |

---

## 11. Triển khai `workflow.py` — Code mẫu đầy đủ

> ⚠️ Đây là code mẫu tham khảo. Bạn cần điều chỉnh theo tool thực tế khám phá được từ `day09 mcp-tools`.

### 11.1. Tạo file helper `src/student_agent/llm_client.py`

```python
"""LLM client wrapper cho model < 10B tham số."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import AsyncOpenAI


def _build_client() -> AsyncOpenAI:
    """Khởi tạo OpenAI-compatible client dựa trên LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        return AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
    elif provider == "ollama":
        return AsyncOpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    elif provider == "together":
        return AsyncOpenAI(
            api_key=os.getenv("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        )
    elif provider == "openai":
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def get_model() -> str:
    return os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


async def ask_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    json_mode: bool = True,
) -> str:
    """Gọi LLM và trả về response text."""
    client = get_client()
    model = get_model()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or "{}"


def parse_json_safe(text: str) -> dict[str, Any]:
    """Parse JSON từ LLM output, xử lý cả trường hợp LLM bọc trong markdown."""
    text = text.strip()
    # Xử lý trường hợp LLM wrap trong ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
```

### 11.2. Triển khai `workflow.py` đầy đủ

Thay thế nội dung `src/student_agent/workflow.py`:

```python
"""L3A Multi-Agent Workflow — sử dụng model < 10B tham số."""
from __future__ import annotations

import json
import re
from typing import Any

from .llm_client import ask_llm, parse_json_safe
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter


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
    message = case.get("customer_message", case.get("message", ""))

    # Trích xuất IDs từ case data
    order_ids = []
    if "order_id" in case:
        order_ids.append(case["order_id"])

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
    all_order_ids = list(set(order_ids + [oid for oid in llm_order_ids if oid]))
    result["order_ids"] = all_order_ids
    return result


# ──────────────────────────────────────────────
#  SPECIALIST AGENTS
# ──────────────────────────────────────────────

async def order_agent(case_id, order_ids, gateway, trace, available_tools):
    """Order Agent: lấy thông tin đơn hàng và items."""
    results = {"orders": [], "items": [], "evidence_refs": []}
    for order_id in order_ids:
        if "get_order" in available_tools:
            try:
                ev = await gateway.call("get_order", case_id=case_id, order_id=order_id)
                results["orders"].append(ev["data"])
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="order-agent", tool_name="get_order",
                           evidence_refs=[ev["evidence_ref"]])
            except Exception:
                pass
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
            except Exception:
                pass
    return results


async def payment_agent(case_id, order_ids, gateway, trace, available_tools):
    """Payment Agent: lấy thông tin thanh toán."""
    results = {"payments": [], "evidence_refs": []}
    for order_id in order_ids:
        if "get_payment" in available_tools:
            try:
                ev = await gateway.call("get_payment", case_id=case_id, order_id=order_id)
                results["payments"].append(ev["data"])
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="payment-agent", tool_name="get_payment",
                           evidence_refs=[ev["evidence_ref"]])
            except Exception:
                pass
    return results


async def shipment_agent(case_id, order_ids, gateway, trace, available_tools):
    """Shipment Agent: lấy thông tin vận chuyển."""
    results = {"shipments": [], "evidence_refs": []}
    for order_id in order_ids:
        if "get_shipment" in available_tools:
            try:
                ev = await gateway.call("get_shipment", case_id=case_id, order_id=order_id)
                results["shipments"].append(ev["data"])
                results["evidence_refs"].append(ev["evidence_ref"])
                trace.emit(case_id=case_id, event_type="tool_result_consumed",
                           actor="shipment-agent", tool_name="get_shipment",
                           evidence_refs=[ev["evidence_ref"]])
            except Exception:
                pass
    return results


async def policy_agent(case_id, gateway, trace, available_tools, issue_type="refund"):
    """Policy Agent: lấy thông tin chính sách."""
    results = {"policies": [], "evidence_refs": []}
    if "get_policy" in available_tools:
        try:
            ev = await gateway.call("get_policy", case_id=case_id, policy_type=issue_type)
            results["policies"].append(ev["data"])
            results["evidence_refs"].append(ev["evidence_ref"])
            trace.emit(case_id=case_id, event_type="tool_result_consumed",
                       actor="policy-agent", tool_name="get_policy",
                       evidence_refs=[ev["evidence_ref"]])
        except Exception:
            pass
    return results


# ──────────────────────────────────────────────
#  ANALYZER: Phân tích tổng hợp bằng LLM
# ──────────────────────────────────────────────

async def analyze_case(case, parsed, order_data, payment_data,
                       shipment_data, policy_data, all_evidence_refs):
    """Dùng LLM để phân tích và đưa ra kết luận."""

    system_prompt = f"""You are an expert e-commerce complaint investigator.
Based on the evidence data, analyze the case and return a JSON with:
{{
  "primary_issue": "<one of: {', '.join(PRIMARY_ISSUES)}>",
  "case_status": "<one of: {', '.join(CASE_STATUSES)}>",
  "confidence": <float 0.0-1.0>,
  "ranked_causes": [{{"cause_code": "<UPPERCASE_CODE>", "rank": 1}}],
  "responsible_parties": [{{"party_type": "<one of: {', '.join(PARTY_TYPES)}>", "party_id": "<id or null>"}}],
  "recommended_refund_brl": <number >= 0>,
  "refund_lines": [{{"reason_code": "<reason>", "amount_brl": <number>, "entity_id": "<id or null>"}}],
  "resolution_actions": ["<action 1>", "<action 2>"],
  "claim_assessments": [{{"claim_id": "<id>", "verdict": "<one of: {', '.join(VERDICTS)}>", "confidence": <float>}}],
  "data_conflicts": []
}}
Rules:
- cause_code: uppercase letters/digits/underscores, 3-80 chars, starts with letter
- If evidence insufficient, use primary_issue="insufficient_evidence"
- Do NOT invent data"""

    user_prompt = f"""Case ID: {case.get('case_id')}
Complaint: {case.get('customer_message', case.get('message', 'N/A'))}
Analysis: {json.dumps(parsed, ensure_ascii=False, default=str)}

=== EVIDENCE ===
Orders: {json.dumps(order_data.get('orders', []), ensure_ascii=False, default=str)}
Items: {json.dumps(order_data.get('items', []), ensure_ascii=False, default=str)}
Payments: {json.dumps(payment_data.get('payments', []), ensure_ascii=False, default=str)}
Shipments: {json.dumps(shipment_data.get('shipments', []), ensure_ascii=False, default=str)}
Policies: {json.dumps(policy_data.get('policies', []), ensure_ascii=False, default=str)}
Evidence refs: {json.dumps(all_evidence_refs)}"""

    return parse_json_safe(await ask_llm(system_prompt, user_prompt))


# ──────────────────────────────────────────────
#  VERIFIER: Kiểm tra và sửa output cho đúng schema
# ──────────────────────────────────────────────

def verify_and_fix(analysis, case_id, all_evidence_refs, affected_entities):
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
        code = c.get("cause_code", "UNKNOWN_CAUSE")
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
    refund = analysis.get("recommended_refund_brl", 0)
    try:
        refund = max(0.0, float(refund))
    except (ValueError, TypeError):
        refund = 0.0

    refund_lines = []
    for line in analysis.get("refund_lines", []):
        amt = line.get("amount_brl", 0)
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
        actions = ["Review case and take appropriate action"]
    actions = actions[:8]

    # claim_assessments
    claims = []
    for cl in analysis.get("claim_assessments", []):
        cid = str(cl.get("claim_id", f"claim_{len(claims)+1}"))[:64]
        v = cl.get("verdict", "insufficient_evidence")
        if v not in VERDICTS:
            v = "insufficient_evidence"
        c = cl.get("confidence", 0.5)
        try:
            c = max(0.0, min(1.0, float(c)))
        except (ValueError, TypeError):
            c = 0.5
        claims.append({"claim_id": cid, "verdict": v, "confidence": c,
                        "evidence_refs": list(dict.fromkeys(all_evidence_refs))[:30]})
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
    available_tools = await gateway.list_tools()

    # Phase 1: Coordinator parses complaint
    parsed = await coordinator_parse(case)
    order_ids = parsed.get("order_ids", [])
    if not order_ids:
        for key in ("order_id", "order_ids"):
            val = case.get(key)
            if isinstance(val, str) and val:
                order_ids.append(val)
            elif isinstance(val, list):
                order_ids.extend([v for v in val if isinstance(v, str) and v])

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
    policy_data = await policy_agent(case_id, gateway, trace, available_tools,
                                     issue_type=parsed.get("complaint_type", "refund"))

    # Phase 3: Collect evidence refs
    all_refs = []
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
                    item_ids.append(str(it[k])); break
            if it.get("seller_id"):
                seller_ids.append(str(it["seller_id"]))
    for p in payment_data.get("payments", []):
        if isinstance(p, dict):
            for k in ("payment_sequential", "payment_id", "payment_type"):
                if k in p and p[k]:
                    pay_refs.append(str(p[k])); break
        elif isinstance(p, list):
            for pp in p:
                if isinstance(pp, dict):
                    for k in ("payment_sequential", "payment_id"):
                        if k in pp and pp[k]:
                            pay_refs.append(str(pp[k])); break
    for s in shipment_data.get("shipments", []):
        if isinstance(s, dict):
            for k in ("shipment_id", "tracking_number"):
                if k in s and s[k]:
                    ship_ids.append(str(s[k])); break

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
                                  shipment_data, policy_data, all_refs)

    # Phase 6: Policy Decision
    trace.emit(case_id=case_id, event_type="policy_decided", actor="coordinator",
               decision_code=analysis.get("primary_issue", "UNKNOWN").upper())

    # Phase 7: Verification
    trace.emit(case_id=case_id, event_type="handoff", actor="coordinator", target="verifier")
    output = verify_and_fix(analysis, case_id, all_refs, entities)
    trace.emit(case_id=case_id, event_type="verification_completed", actor="verifier")

    return output
```

### 11.3. Cài thêm dependency

```bash
pip install openai
```

---

## 12. Cập nhật `ARCHITECTURE.md`

Sửa file `ARCHITECTURE.md` theo thiết kế thực tế của bạn (xem phần 10).

Cần điền đầy đủ tất cả TODO trong file gốc, bao gồm:
- System overview (sơ đồ luồng)
- Agent ownership (bảng vai trò)
- A2A protocol (cách agents giao tiếp)
- Evidence lifecycle
- Failure policy (bảng xử lý lỗi)
- Verification invariants
- Reproducibility (model, config, lệnh chạy)

---

## 13. Chạy, validate & debug

### 13.1. Chạy workflow

```bash
day09 run
```

### 13.2. Validate kết quả

```bash
day09 validate
```

Output mong đợi:
```
OK: 100 outputs / <N> trace events
```

### 13.3. Debug tips

**Nếu MCP tool lỗi:**
```bash
day09 mcp-tools
```

**Nếu LLM trả JSON sai:**
- Giảm `temperature` xuống 0.0
- Thêm ví dụ cụ thể trong prompt
- Dùng `response_format={"type": "json_object"}`

**Chạy thử 1 case:**
```python
# scratch_test.py
import asyncio, json
from student_agent.config import Settings
from student_agent.contracts import Contracts
from student_agent.mcp_gateway import connect_gateway
from student_agent.trace import TraceWriter

async def test_one():
    settings = Settings.load()
    contracts = Contracts(settings.root / "contracts" / "schemas")
    trace = TraceWriter(settings.root / "traces" / "trace_test.jsonl", contracts)
    case = json.loads((settings.root / "inputs" / "L3A_CASE_001.json").read_text())
    print(f"Case: {json.dumps(case, indent=2, ensure_ascii=False)}")
    async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gw:
        tools = await gw.list_tools()
        print(f"Tools: {tools}")

asyncio.run(test_one())
```

---

## 14. Đóng gói & nộp bài

### 14.1. Package

```bash
day09 package --output dist/submission.zip
```

### 14.2. ZIP chỉ được chứa
```
manifest.json
trace.jsonl
outputs/<case_id>.json
```

> ⚠️ KHÔNG đưa source, input, `.env`, API key vào ZIP.

### 14.3. Upload
Upload `dist/submission.zip` tại workspace `/l3a`.

### 14.4. Git
```bash
git add .
git commit -m "Implement multi-agent workflow"
git push origin DinhDucThai
```

---

## 15. Tiêu chí chấm điểm & chiến lược tối ưu

| Thành phần | Trọng số | Chiến lược |
|-----------|-------:|-----------|
| **Semantic** | **45%** | Đảm bảo `primary_issue` đúng, `resolution_actions` hợp lý |
| **Evidence** | **15%** | Thu thập đầy đủ, chỉ trích dẫn liên quan |
| **Provenance** | **15%** | KHÔNG tự tạo `evidence_ref` |
| **Consistency** | **10%** | Status/refund/action logic nhất quán |
| **Schema** | **5%** | Verifier đã handle |
| **Calibration** | **5%** | Confidence phản ánh thực tế |
| **Workflow** | **5%** | Đủ lifecycle events, đúng thứ tự |
| **Efficiency** | **0%** | Không tính nhưng vẫn audit |

---

## 16. Lưu ý quan trọng & sai lầm phổ biến

### ❌ Sai lầm thường gặp

| # | Sai lầm | Hậu quả |
|---|---------|---------|
| 1 | Tự bịa `evidence_ref` | 0 điểm provenance |
| 2 | Dùng evidence chéo case | 0 điểm provenance |
| 3 | Commit `.env` lên Git | Lộ key |
| 4 | Output thiếu required fields | 0 điểm schema |
| 5 | `case_id` không khớp | 0 điểm toàn case |
| 6 | Dùng model > 10B | Vi phạm yêu cầu |
| 7 | Thiếu trace events | Mất điểm workflow |
| 8 | `confidence` luôn = 1.0 | Mất điểm calibration |

### ✅ Checklist trước khi nộp

- [ ] `day09 validate` pass
- [ ] Model < 10B tham số
- [ ] Không có API key trong code/output
- [ ] Trace đầy đủ lifecycle events
- [ ] Output có `evidence_refs` thật
- [ ] `ARCHITECTURE.md` đã cập nhật
- [ ] `day09 package` thành công
- [ ] Upload ZIP tại `/l3a`
- [ ] Push code lên Git

---

## Tóm tắt lệnh

```bash
# Setup
python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pip install openai

# Verify
pytest -q && day09 --help

# Discover tools
day09 mcp-tools

# Run & validate
day09 run
day09 validate

# Package & submit
day09 package --output dist/submission.zip

# Git
git add . && git commit -m "Implement workflow" && git push origin DinhDucThai
```
