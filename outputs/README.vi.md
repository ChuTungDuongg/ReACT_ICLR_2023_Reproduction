# Kết quả benchmark runtime

> 🌐 **Ngôn ngữ:** [English](README.md) | Tiếng Việt

`outputs/` chứa kết quả sinh ra khi chạy benchmark. Nội dung runtime trong thư
mục này bị Git ignore; chỉ `README.md`, `README.vi.md` và `.gitkeep` nên được
version control.

## Cấu trúc một run

```text
outputs/<task>/<method>/<UTC timestamp>/
├── config.json
├── metrics.json
├── predictions.jsonl
├── trajectories.jsonl   # Có ở CoT-SC, Act-only, ReAct và hybrid
└── run.log
```

## Chức năng từng file

- `config.json`: lưu toàn bộ cấu hình cần để hiểu/tái hiện run, gồm `batch_size`.
- `predictions.jsonl`: mỗi example một dòng, được flush theo thứ tự dataset sau
  khi batch tương ứng hoàn thành.
- `trajectories.jsonl`: mỗi CoT-SC/Act/ReAct/hybrid example một dòng, chứa các bước.
- `metrics.json`: đủ 12 official answer/supporting-fact/joint HotpotQA metrics,
  evidence coverage, runtime, steps, tool calls và lý do dừng.
- `run.log`: bản lưu của progress và trajectory đã in live trên stdout.

Toàn bộ final metrics cũng được in ra terminal/Colab nếu không dùng `--quiet`.
Khi agent không xuất cặp evidence `(title, sentence_id)`, SP/joint metrics chấm
prediction rỗng và evidence coverage thể hiện rõ giới hạn đó.

JSONL được dùng vì có thể append từng record. Nếu run bị ngắt giữa chừng, các
example đã hoàn thành vẫn còn trên disk thay vì mất toàn bộ kết quả.

Với `react-cot-sc` và `cot-sc-react`, prediction còn lưu vote count/confidence,
nhánh được chọn và trạng thái fallback. Mỗi bước hybrid trajectory có field
`phase` bằng `react` hoặc `cot_sc`.
Nếu run batch bị ngắt, chỉ batch đang thực thi có thể chưa được ghi; mọi batch
đã hoàn thành trước đó vẫn còn trên disk.

## Lưu ý

- Không đặt source code hoặc tài liệu chỉnh tay trong `outputs/`.
- Không commit benchmark output lớn, model, cache, dataset hoặc secret.
- Tài liệu ổn định cần chia sẻ phải đặt trong `output/` (không có chữ `s`).
- Smoke output không tự động trở thành research result; luôn ghi rõ model, số
  mẫu và mục tiêu của run.
