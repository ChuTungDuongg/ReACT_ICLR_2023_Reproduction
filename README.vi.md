# 🧠 Tái hiện paper ReAct

> 🌐 **Ngôn ngữ:** [English](README.md) | Tiếng Việt

<p align="center">
  <img src="https://img.shields.io/badge/Paper-ICLR%202023-6f42c1" alt="Paper ICLR 2023">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Trạng_thái-Hoàn%20tất%20Sprint%204-238636" alt="Hoàn tất Sprint 4">
  <img src="https://img.shields.io/badge/Tests-48%20PASS-18A0AE" alt="48 tests PASS">
  <img src="https://img.shields.io/badge/Điểm_vào-main.py-102A43" alt="Chạy bằng main.py">
</p>

<p align="center">
  <a href="output/pdf/react-reproduction-roadmap.pdf">📄 Roadmap PDF</a> •
  <a href="#chạy-trên-google-colab">☁️ Google Colab</a> •
  <a href="#cách-hoạt-động">🧩 Cách hoạt động</a> •
  <a href="#cấu-trúc-repository">🗂️ Cấu trúc file</a> •
  <a href="#kiểm-thử">🧪 Kiểm thử</a>
</p>

Đây là repository tái hiện các phương pháp **Standard**, **Chain-of-Thought
(CoT)**, **Act-only** và **ReAct** trên HotpotQA. Model được chạy bằng Hugging
Face; Act-only và ReAct có thể tìm kiếm Wikipedia thật. Toàn bộ chương trình có
một điểm vào duy nhất là `main.py`.

Cả bốn phương pháp hiện dùng đúng bộ 6 ví dụ HotpotQA viết thủ công trong
Appendix C.1 của paper. CLI mặc định chọn `Qwen/Qwen2.5-7B-Instruct`; vẫn có thể
dùng `--model` để chọn model causal LM tương thích khác.

> Đây là reproduction study dùng modern instruction-tuned LLM. Nó không phải
> bản tái hiện chính xác PaLM-540B của paper gốc, vì model, hạ tầng và trạng thái
> Wikipedia đều khác.

## ✅ Trạng thái hiện tại

Repository đã hoàn tất **Sprint 0 đến Sprint 4** và dừng đúng tại Sprint 4.
Sprint 5, FEVER, ALFWorld, WebShop và interactive app chưa được triển khai.

| Sprint | Nội dung | Trạng thái |
|---:|---|---|
| 0 | Khung repository, config, logging, CLI | ✅ Hoàn tất |
| 1 | HotpotQA, metrics, schema và experiment runner | ✅ Hoàn tất |
| 2 | Hugging Face LLM, Standard và CoT | ✅ Hoàn tất |
| 3 | Wikipedia environment và Act-only | ✅ Hoàn tất |
| 4 | ReAct, live trajectory và lưu trajectory | ✅ Hoàn tất |
| 5+ | So sánh lớn, FEVER, phân tích và phần mở rộng | ⏸️ Chưa bắt đầu |

Roadmap, requirements và acceptance criteria chi tiết nằm trong
[roadmap PDF](output/pdf/react-reproduction-roadmap.pdf).

## 🚀 Cài đặt nhanh

```bash
git clone https://github.com/ChuTungDuongg/ReACT_ICLR_2023_Reproduction.git
cd ReACT_ICLR_2023_Reproduction
python -m venv .venv
```

Kích hoạt môi trường:

```bash
# Linux hoặc macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Cài dependency và kiểm tra project:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --help
python main.py doctor
```

Repository không dùng `.ipynb`. Dataset và model được tải ở lần chạy đầu tiên
vào `cache/huggingface/`; thư mục này đã được Git ignore.

<a id="chạy-trên-google-colab"></a>

## ☁️ Chạy trên Google Colab

Chạy cell cài đặt:

```python
!git clone https://github.com/ChuTungDuongg/ReACT_ICLR_2023_Reproduction.git
%cd ReACT_ICLR_2023_Reproduction
!python -m pip install -q -r requirements.txt
!python main.py doctor
```

Sau đó chạy ReAct với 5 mẫu HotpotQA:

```python
!python main.py benchmark \
    --task hotpotqa \
    --method react \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --show-trajectories
```

### Ghi chú Sprint 4 khi chạy HotpotQA trên Colab

- `--num-samples` là tổng số example, không phải batch size. Sprint 4 chạy tuần
  tự từng example; model chỉ load một lần rồi được tái sử dụng.
- Split hiện tại `hotpotqa/hotpot_qa`, `distractor`, `validation` có tối đa
  **7.405 examples**. Giá trị hợp lệ là `1-7405`; nhập lớn hơn sẽ dừng với lỗi
  validation.
- Nên dùng 5 mẫu để smoke test, 20-50 mẫu để kiểm tra prompt/failure, và chỉ
  chạy 100/500+ sau khi đã xác nhận runtime cùng độ ổn định của MediaWiki.
- Không thêm `--quiet` trên Colab. Metrics của từng example, toàn bộ 12 official
  HotpotQA metrics và operational metrics cuối run sẽ hiện trên cell, đồng thời
  được lưu vào `run.log`.
- `metrics.json` chứa answer `EM/F1/precision/recall`, supporting-fact
  `SP EM/F1/precision/recall`, joint `EM/F1/precision/recall`, evidence coverage,
  runtime, steps/tools trung bình và termination reasons.
- Công thức bám theo
  [official HotpotQA evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
- Agent Sprint 4 hiện trả lời answer nhưng chưa xuất cặp evidence HotpotQA
  `(title, sentence_id)`. Evaluator chính thức sẽ chấm supporting-fact prediction
  rỗng (SP/joint thường bằng 0), báo rõ evidence coverage và không tự bịa evidence.
- Qwen2.5-7B cần nhiều bộ nhớ hơn đáng kể so với bản 3B trước đây. Nên chọn GPU
  Colab loại L4/A100. T4 có thể phải offload một phần model sang RAM nhờ
  `device_map=auto`, vì vậy tốc độ sẽ chậm hơn nhiều.

Chương trình tự chọn CUDA nếu PyTorch nhìn thấy GPU; nếu không có CUDA thì tự
chuyển sang CPU. `bitsandbytes` không bắt buộc. Có thể khai báo `HF_TOKEN` trong
Colab để tăng giới hạn tải từ Hugging Face Hub.

## ⌨️ Các lệnh chính

Kiểm tra CLI và cấu hình:

```bash
python main.py --help
python main.py benchmark --help
python main.py doctor
```

Chạy Standard:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method standard \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-samples 5 \
    --seed 42
```

Thay `standard` bằng `cot`, `act` hoặc `react` để đổi phương pháp. Với Act-only
và ReAct, nên thêm `--show-trajectories` để xem trực tiếp Action/Observation và
Thought nếu có.

`--model` là tùy chọn; nếu bỏ qua, CLI tự dùng
`Qwen/Qwen2.5-7B-Instruct`. Standard, CoT, Act-only và ReAct nhận prompt pack
6-shot tương ứng trong Appendix C.1; example cần chấm vẫn chỉ đưa câu hỏi vào
model, không đưa supporting paragraphs.

Các tham số hữu ích:

| Tham số | Ý nghĩa |
|---|---|
| `--device auto` | Tự chọn CUDA, MPS hoặc CPU |
| `--num-samples 5` | Số câu hỏi được chạy |
| `--seed 42` | Cố định cách lấy mẫu và random seed |
| `--max-agent-steps 7` | Số bước tối đa của Act/ReAct |
| `--max-new-tokens 256` | Số token sinh tối đa trong một lượt |
| `--show-trajectories` | In từng Thought/Action/Observation |
| `--log-level INFO` | Mức chi tiết của log |

<a id="cách-hoạt-động"></a>

## 🧩 Bốn phương pháp hoạt động như thế nào?

| Phương pháp | Có reasoning rõ ràng? | Dùng Wikipedia? | Luồng xử lý |
|---|---:|---:|---|
| Standard | Không | Không | Câu hỏi → câu trả lời |
| CoT | Có | Không | Câu hỏi → suy luận → câu trả lời |
| Act-only | Không | Có | Action ↔ Observation → Finish |
| ReAct | Có | Có | Thought → Action → Observation → Finish |

Giải thích ngắn:

- **Standard** yêu cầu model trả lời trực tiếp từ kiến thức đã học.
- **CoT** yêu cầu model trình bày reasoning trước khi đưa ra đáp án.
- **Act-only** không lưu Thought; model chỉ chọn hành động Wikipedia.
- **ReAct** kết hợp reasoning và hành động. Observation luôn do environment
  tạo ra, không tin Observation do model tự viết.

Core agent được viết trực tiếp bằng Python, không dùng LangChain, LangGraph,
CrewAI hay AutoGen. Nhờ vậy có thể đọc rõ parser, state, vòng lặp và nguyên nhân
dừng của từng example.

## 🏗️ Luồng chạy tổng thể

```text
main.py
  → CLI đọc tham số và config
  → HotpotQA loader lấy mẫu theo seed
  → HuggingFaceProvider load tokenizer/model
  → agent Standard / CoT / Act-only / ReAct
       └─ Act/ReAct → WikipediaEnvironment → MediaWiki API
  → evaluator chính thức answer/supporting-fact/joint của HotpotQA
  → lưu config, prediction, trajectory, metrics và log
```

<a id="cấu-trúc-repository"></a>

## 🗂️ Cấu trúc repository

```text
.
├── main.py                         # Điểm chạy duy nhất
├── requirements.txt                # Dependency runtime và test
├── configs/                        # Cấu hình mặc định có version control
├── output/                         # Tài liệu ổn định, ví dụ roadmap PDF
├── outputs/                        # Kết quả benchmark runtime, bị Git ignore
├── src/react_reproduction/
│   ├── agents/                     # Standard, CoT, Act-only, ReAct và parser
│   ├── datasets/                   # BenchmarkExample và HotpotQA loader
│   ├── evaluation/                 # Full official metrics và schema kết quả
│   ├── experiments/                # Runner và ghi artifact
│   ├── llm/                        # Interface LLM và Hugging Face provider
│   ├── prompts/                    # Prompt cho từng phương pháp
│   ├── tools/                      # Wikipedia client/environment
│   ├── cli.py                      # Khai báo command và kết nối component
│   ├── config.py                   # Đọc, kiểm tra YAML và biến môi trường
│   └── logging_utils.py            # Live stdout UTF-8 và run.log
└── tests/                           # 48 unit/integration-style tests
```

README tiếng Việt của từng thư mục:

- [Giải thích source code](src/README.vi.md)
- [Giải thích configuration](configs/README.vi.md)
- [Giải thích tests](tests/README.vi.md)
- [Giải thích benchmark outputs](outputs/README.vi.md)
- [Giải thích project documents](output/README.vi.md)

## 📦 Kết quả của mỗi lần chạy

Mỗi benchmark tạo một thư mục riêng theo thời gian UTC:

```text
outputs/hotpotqa/react/<UTC timestamp>/
├── config.json
├── metrics.json
├── predictions.jsonl
├── trajectories.jsonl
└── run.log
```

| File | Chức năng |
|---|---|
| `config.json` | Lưu model, dataset, method, seed, generation config và device |
| `predictions.jsonl` | Mỗi dòng là kết quả của một example; flush ngay sau example |
| `trajectories.jsonl` | Mỗi dòng là toàn bộ trajectory của một Act/ReAct example |
| `metrics.json` | Full official HotpotQA metrics, runtime, steps/tool calls và lý do dừng |
| `run.log` | Bản log được lưu song song với stdout |

`--show-trajectories` làm stdout hiển thị từng Thought, Action, Observation và
raw model output để dễ debug trên terminal hoặc ngay trong cell Colab.

## 📊 Smoke benchmark đã xác minh

Đây chỉ là kiểm tra pipeline nhỏ, không phải benchmark dùng để kết luận chất
lượng model.

| Task | Method/model | Samples | Exact Match | Lý do dừng |
|---|---|---:|---:|---|
| HotpotQA | ReAct / Qwen2.5-0.5B-Instruct | 3 | 0.0 (0/3) | loop 1, max-step 1, parse 1 |

Run này dùng inference Hugging Face thật, MediaWiki thật và CPU cục bộ. Cả ba
example đều có prediction/trajectory được lưu. Model 0.5B chưa tạo được đáp án
`Finish` đúng; repository không báo cáo kết quả tốt hơn thực tế.

<a id="kiểm-thử"></a>

## 🧪 Kiểm thử

```bash
python -m pytest -q
python -m compileall -q main.py src tests
python main.py --help
python main.py doctor
```

48 tests hiện tại kiểm tra:

- CLI, cấu hình và `doctor`;
- HotpotQA loading, validation và sampling theo seed;
- answer EM/F1/precision/recall, supporting-fact và joint metrics chính thức;
- 6-shot prompt pack của paper, nhãn đánh số và Standard/CoT parser;
- Search/Lookup/Finish, trang thiếu hoặc mơ hồ;
- Lookup nhiều lần, loop detection và max-step;
- Act-only/ReAct, parsing recovery và trajectory history;
- flush `predictions.jsonl` và `trajectories.jsonl` theo từng example.

Unit tests không tải model và không gọi Wikipedia thật. Chúng sử dụng scripted
LLM, dữ liệu inject và fake Wikipedia client nên chạy nhanh, ổn định.

## ⚙️ Dependency chính

- `torch`, `transformers`, `accelerate`: chạy model Hugging Face và tự đặt
  device;
- `datasets`: tải/cache HotpotQA;
- `requests`, `tenacity`: gọi MediaWiki có timeout và retry;
- `PyYAML`, `python-dotenv`: cấu hình;
- `pytest`: chạy test repository.

Model được load lazy, chỉ tải khi chạy benchmark. CUDA ưu tiên BF16 nếu GPU hỗ
trợ, nếu không dùng FP16; CPU dùng FP32.

## 📖 Nếu đây là lần đầu bạn implement kiểu project này

Nên đọc theo thứ tự sau:

1. Đọc file này để hiểu mục tiêu và luồng tổng thể.
2. Đọc [src/README.vi.md](src/README.vi.md) để biết từng package chịu trách
   nhiệm gì.
3. Mở `main.py`, sau đó theo luồng sang `cli.py`.
4. Đọc `agents/react.py` cùng `tools/wikipedia.py` để hiểu vòng lặp ReAct.
5. Đọc `experiments/runner.py` và `artifacts.py` để hiểu cách lưu kết quả.
6. Đọc [tests/README.vi.md](tests/README.vi.md), rồi chạy từng test file khi
   muốn sửa một component.

## ⚠️ Giới hạn

- Milestone hiện tại dừng ở Sprint 4; chưa có benchmark so sánh quy mô lớn.
- Không thể kỳ vọng tái hiện chính xác số liệu PaLM-540B của paper gốc.
- Wikipedia và kết quả search có thể thay đổi theo thời gian.
- Model 3B có thể chậm hoặc thiếu RAM/VRAM trên máy cá nhân; Colab GPU phù hợp
  hơn cho command mục tiêu.
- Model đôi khi sinh sai format. Parser ghi nhận lỗi, lấy Action hợp lệ đầu tiên,
  còn environment chặn loop và giới hạn số bước.

## 📄 Trích dẫn paper

```bibtex
@inproceedings{yao2023react,
  title     = {ReAct: Synergizing Reasoning and Acting in Language Models},
  author    = {Shunyu Yao and Jeffrey Zhao and Dian Yu and Nan Du and
               Izhak Shafran and Karthik Narasimhan and Yuan Cao},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```
