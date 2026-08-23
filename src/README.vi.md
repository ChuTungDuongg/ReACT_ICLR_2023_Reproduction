# Source code

> 🌐 **Ngôn ngữ:** [English](README.md) | Tiếng Việt

Thư mục `src/` chứa toàn bộ logic chính của chương trình. Project dùng kiểu
layout chuẩn `src`: package thật nằm tại `src/react_reproduction/`. `main.py` ở
root tự thêm `src/` vào Python path nên sau khi clone và cài requirements có thể
chạy ngay, không cần `pip install -e .`.

## Bản đồ package

| Đường dẫn | Chức năng | Hoạt động ra sao |
|---|---|---|
| `react_reproduction/cli.py` | Command-line interface | Đọc tham số, config, tạo model/agent/environment rồi gọi runner |
| `react_reproduction/config.py` | Cấu hình typed | Đọc YAML, biến môi trường, kiểm tra giá trị và resolve đường dẫn |
| `react_reproduction/logging_utils.py` | Logging | In UTF-8 live ra stdout và đồng thời ghi `run.log` |
| `react_reproduction/datasets/` | Dữ liệu benchmark | Chuẩn hóa raw HotpotQA thành `BenchmarkExample`, lấy mẫu theo seed |
| `react_reproduction/llm/` | Model provider | Định nghĩa interface chung và implementation Hugging Face |
| `react_reproduction/prompts/` | Prompt builder | Dựng Standard, CoT, Act-only và ReAct từ 6 ví dụ HotpotQA viết thủ công trong Appendix C.1 của paper |
| `react_reproduction/agents/` | Logic phương pháp | Chạy model, parse output, quản lý vòng lặp và tạo trajectory |
| `react_reproduction/tools/` | Wikipedia tool | Search/Lookup/Finish, state bài viết, timeout/retry, loop/max-step |
| `react_reproduction/evaluation/` | Chấm điểm | Tính official answer/supporting-fact/joint metrics và schema JSON |
| `react_reproduction/experiments/` | Điều phối run | Chạy từng example, đo latency, flush JSONL và ghi metrics |

## Luồng gọi code

```text
main.py
  → cli.main()
  → load_project_config()
  → load_hotpotqa()
  → HuggingFaceProvider
  → StandardAgent / CoTAgent / ActOnlyAgent / ReActAgent
  → run_hotpotqa_benchmark()
  → outputs/<task>/<method>/<timestamp>/
```

## `agents/` có gì?

| File | Vai trò |
|---|---|
| `base.py` | `BaseAgent`, `AgentResult`, `TrajectoryStep` dùng chung |
| `parsing.py` | Parse final answer, reasoning và Search/Lookup/Finish |
| `standard.py` | Gọi model một lần và lấy đáp án trực tiếp |
| `cot.py` | Gọi model một lần, lưu reasoning rồi lấy final answer |
| `act.py` | Lặp Action → Observation, không lưu Thought |
| `react.py` | Lặp Thought → Action → Observation đến khi Finish/dừng |

## `tools/wikipedia.py` quản lý state như thế nào?

- `Search[entity]` tìm và mở một bài Wikipedia.
- Bài đang mở được lưu trong `current_article`.
- `Lookup[text]` tìm câu khớp tiếp theo trong bài đang mở.
- Offset của từng Lookup được lưu để lần gọi sau trả về kết quả tiếp theo.
- `Finish[answer]` kết thúc example.
- `Search` trùng lần hai không gọi Wikipedia lại mà trả hướng dẫn chuyển sang
  `Lookup`, một entity khác hoặc `Finish`; lần ba mới ghi nhận `action_loop`.
- `Lookup` giống nhau vẫn được phép lặp để lấy câu khớp tiếp theo.
- Nếu gặp `action_loop` khi còn budget, agent dành lượt kế tiếp để tạo
  `Finish[best answer]`; bước cuối của budget cũng luôn dành cho `Finish`.
- Hết budget bị chặn bằng `max_steps_exceeded`.

## Quy tắc khi sửa source

- Không đặt network/model download ở import time.
- Giữ agent loop trực tiếp, dễ đọc và không phụ thuộc agent framework.
- Component mới phải có test bằng fake/scripted dependency trước khi chạy thật.
- Luôn giữ seed và config trong artifact để có thể tái hiện run.
- Không triển khai feature của sprint chưa được cho phép.

Implementation hiện hoàn tất đến Sprint 4. Model mặc định của CLI là
`Qwen/Qwen2.5-7B-Instruct`.
