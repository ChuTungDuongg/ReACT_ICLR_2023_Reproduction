# Tái hiện paper ReAct

> Ngôn ngữ: [English](README.md) | Tiếng Việt

Repository triển khai có thể kiểm tra được cho bảy phương pháp trong paper
"ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023):
Standard, CoT, CoT-SC, Act, ReAct, CoT-SC -> ReAct và ReAct -> CoT-SC.
HotpotQA và FEVER dùng chung CLI, model provider, batch runner, Wikipedia
environment, trajectory, logging và quy ước output.

Paper gốc dùng PaLM-540B. Project hiện dùng
`Qwen/Qwen2.5-7B-Instruct`; đây là tái hiện phương pháp và protocol, không phải
tái hiện chính xác điểm số tuyệt đối của PaLM.

## Trạng thái project

| Sprint | Phạm vi | Trạng thái |
|---:|---|---|
| 0 | Repository, config, logging, CLI | Hoàn tất |
| 1 | HotpotQA dataset, evaluation, artifacts | Hoàn tất |
| 2 | Hugging Face, Standard, CoT | Hoàn tất |
| 3 | Wikipedia environment, Act | Hoàn tất |
| 4 | ReAct và trajectories | Hoàn tất |
| 5 | So sánh bảy method HotpotQA | Hoàn tất về chức năng; benchmark 500 mẫu đã chạy |
| 6 | FEVER bám sát paper | Hoàn tất |
| 7 | Research report và failure analysis | Đã bắt đầu một phần |
| 8 | UI | Chưa bắt đầu |
| 9 | Extension/fine-tuning tùy chọn | Chưa bắt đầu |

HotpotQA đã có kết quả 500 mẫu cho cả bảy method. Các historical run dùng code
version và batch policy khác nhau, vì vậy controlled rerun vẫn là future cleanup.
Phân tích hiện có gồm termination, hybrid path, CoT-SC confidence và overlap.

FEVER đã implement và test; kết quả benchmark 500 mẫu vẫn là **TBD** cho tới
khi chạy thật trên Colab. ALFWorld, WebShop và UI chưa được implement.

Roadmap được duy trì ở [source Markdown](output/react-reproduction-roadmap.md)
và [PDF](output/pdf/react-reproduction-roadmap.pdf).

## Task và method hỗ trợ

Task: `hotpotqa`, `fever`.

Method: `standard`, `cot`, `cot-sc`, `act`, `react`, `cot-sc-react`,
`react-cot-sc`.

Tất cả đều chạy qua `python main.py benchmark`; project không có executable
riêng cho FEVER.

## Protocol FEVER

FEVER là bài toán xác minh claim với đúng ba nhãn: `SUPPORTS`, `REFUTES`,
`NOT ENOUGH INFO`. Model chỉ nhận claim; gold evidence, context và supporting
pages không bao giờ đi vào prompt.

Prompt dùng đúng ba ví dụ thủ công trong Appendix C.2:

1. Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. (`SUPPORTS`)
2. Stranger Things is set in Bloomington, Indiana. (`REFUTES`)
3. Beautiful reached number two on the Billboard Hot 100 in 2003. (`NOT ENOUGH INFO`)

Standard, CoT, Act và ReAct là các ablation có kiểm soát trên cùng ba ví dụ.
Act/ReAct giữ action space `Search[entity]`, `Lookup[string]`, `Finish[label]`.
Search trả năm câu đầu nếu resolve được trang chính xác; nếu không, nó trả tối
đa năm gợi ý Wikipedia. Lookup lặp lại sẽ tiến tới occurrence tiếp theo.

Metric chính là Accuracy. ReAct FEVER mặc định tối đa 5 bước. CoT-SC tạo 21
sample stochastic ở temperature 0.7. Nếu hòa phiếu, implementation chọn nhãn
hợp lệ xuất hiện sớm nhất; đây là chi tiết deterministic của reproduction vì
paper không định nghĩa tie-break. CoT-SC -> ReAct fallback khi winner dưới
`n/2` (tối đa 10/21); ReAct -> CoT-SC chỉ fallback khi ReAct không trả nhãn hợp
lệ trong budget. Sprint 6 không fine-tune.

## Paper và reproduction

| Paper | Reproduction |
|---|---|
| PaLM-540B | Qwen2.5-7B-Instruct |
| FEVER claim-only | Giống paper |
| 3 FEVER exemplars | Đúng Appendix C.2 |
| Search/Lookup/Finish | Cùng action space khái niệm |
| CoT-SC `n=21` | Giống paper |
| CoT-SC temperature `0.7` | Giống paper |
| ReAct FEVER max steps `5` | Giống paper |
| Metric chính Accuracy | Giống paper |
| Fine-tuning | Không thuộc Sprint 6 |
| Batching | Tối ưu kỹ thuật, state mỗi sample độc lập |

Điểm FEVER Accuracy trong Table 1 chỉ là tham chiếu của paper, không phải kết
quả project: Standard 57.1, CoT 56.3, CoT-SC 60.4, Act 58.9, ReAct 60.9,
CoT-SC -> ReAct 64.6, ReAct -> CoT-SC 62.0.

## Cài đặt

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --help
python main.py doctor
```

Model và dataset được cache trong `cache/huggingface/`, thư mục đã bị Git
ignore. Có thể đặt `HF_TOKEN` trong environment để tăng rate limit; không lưu
token vào repository.

## Bảy lệnh FEVER cho Colab

Standard:

```bash
python main.py benchmark \
  --task fever \
  --method standard \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 32
```

CoT:

```bash
python main.py benchmark \
  --task fever \
  --method cot \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 16
```

CoT-SC:

```bash
python main.py benchmark \
  --task fever \
  --method cot-sc \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 16 \
  --cot-sc-samples 21 \
  --cot-sc-temperature 0.7
```

Act:

```bash
python main.py benchmark \
  --task fever \
  --method act \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 16 \
  --max-agent-steps 5
```

ReAct:

```bash
python main.py benchmark \
  --task fever \
  --method react \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 8 \
  --max-agent-steps 5
```

CoT-SC -> ReAct:

```bash
python main.py benchmark \
  --task fever \
  --method cot-sc-react \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 8 \
  --max-agent-steps 5 \
  --cot-sc-samples 21 \
  --cot-sc-temperature 0.7
```

ReAct -> CoT-SC:

```bash
python main.py benchmark \
  --task fever \
  --method react-cot-sc \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 8 \
  --max-agent-steps 5 \
  --cot-sc-samples 21 \
  --cot-sc-temperature 0.7
```

Batch size là engineering configuration, không phải paper setting. Nên chạy
3-5 mẫu trước. `--show-trajectories` hiện toàn bộ reasoning/tool trace; mặc định
CoT-SC chỉ log winner, số phiếu và vote map, còn full samples vẫn được lưu.

## Kiến trúc

```text
main.py / CLI
  -> task dataset adapter (HotpotQA hoặc FEVER)
  -> shared agents cho 7 methods
  -> shared Hugging Face provider và batching
  -> WikipediaEnvironment riêng cho từng sample Act/ReAct
  -> task evaluator (HotpotQA metrics hoặc FEVER Accuracy)
  -> shared serializer và live/file logging
```

Batching không trộn state: mỗi sample giữ current page, Lookup offsets, history,
counters và termination riêng. Hybrid chỉ chạy fallback cho sample thỏa đúng
heuristic của paper.

## Output

```text
outputs/<task>/<method>/<run_id>/
  config.json
  metrics.json
  predictions.jsonl
  trajectories.jsonl
  run.log
```

FEVER mặc định dùng `data/paper_dev.jsonl` từ official ReAct repository và pin
Git commit. Nguồn này đúng với official wrapper, đồng thời tránh dataset script
Hugging Face đã lỗi thời và các dòng claim bị nhân theo evidence trong
`labelled_dev`; source được pin có 9.999 claims.

`config.json` lưu dataset/split, model revision nếu backend cung cấp, method,
seed, sample IDs, batch/generation settings, step budget, CoT-SC settings và
threshold, prompt version, code version, timestamp. FEVER `metrics.json` lưu
Accuracy, invalid count, phân bố nhãn, per-class accuracy, confusion matrix,
average steps/tool calls, runtime và termination counts. Sprint 6 không tính
official FEVER evidence score và không đưa gold evidence vào model.

## Kiểm thử

```bash
python -m pytest -q
python -m compileall -q main.py src tests
python main.py --help
python main.py doctor
```

Test suite offline dùng scripted provider và fake Wikipedia, không tải Qwen.
Nó kiểm tra parser nhãn, deterministic claim-only loader, ba exemplars và bốn
ablation, bảy execution path, biên 11/21 và 10/21, batch isolation,
serialization, Accuracy, paper-fidelity audit và HotpotQA regression.

## Giới hạn

- Kết quả FEVER 500 mẫu là TBD; README không dựng số liệu giả.
- Qwen2.5-7B và Wikipedia hiện tại khác PaLM-540B và snapshot lịch sử.
- MediaWiki thay đổi hoặc gián đoạn có thể ảnh hưởng Act/ReAct.
- Historical HotpotQA runs chưa hoàn toàn controlled.
- HotpotQA giữ repeated-Search loop guard cũ để bảo toàn behavior lịch sử;
  FEVER tắt guard này và cho ReAct dùng hết budget 5 bước như paper.
- HotpotQA chưa xuất `(title, sentence_id)` evidence pairs.
- Fine-tuning, ALFWorld, WebShop và UI nằm ngoài Sprint 6.
