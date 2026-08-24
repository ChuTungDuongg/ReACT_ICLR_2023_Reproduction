# 🧠 Tái hiện paper ReAct

> 🌐 **Ngôn ngữ:** [English](README.md) | Tiếng Việt

<p align="center">
  <img src="https://img.shields.io/badge/Paper-ICLR%202023-6f42c1" alt="Paper ICLR 2023">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Trạng_thái-Sprint%206-238636" alt="Sprint 6 hoàn tất">
  <img src="https://img.shields.io/badge/Tests-146%20PASS-18A0AE" alt="146 tests PASS">
  <img src="https://img.shields.io/badge/Điểm_vào-main.py-102A43" alt="Chạy bằng main.py">
</p>

<p align="center">
  <a href="output/pdf/react-reproduction-roadmap.pdf">📄 Roadmap PDF</a> •
  <a href="#chạy-trên-google-colab">☁️ Google Colab</a> •
  <a href="#cách-hoạt-động">🧩 Cách hoạt động</a> •
  <a href="#cấu-trúc-repository">🗂️ Cấu trúc file</a> •
  <a href="#kiểm-thử">🧪 Kiểm thử</a>
</p>

Đây là repository tái hiện **Standard**, **Chain-of-Thought**, **CoT-SC**,
**Act-only**, **ReAct**, **ReAct → CoT-SC** và **CoT-SC → ReAct** trên HotpotQA
và FEVER. Model chạy bằng Hugging Face; các nhánh ReAct có thể tìm kiếm
Wikipedia thật. Toàn bộ chương trình có một điểm vào duy nhất là `main.py`.

Tất cả phương pháp tái sử dụng ví dụ từ paper: 6 ví dụ HotpotQA trong Appendix
C.1 hoặc 3 ví dụ FEVER trong Appendix C.2. CoT-SC mặc định dùng 21 samples,
temperature 0.7. CLI mặc định chọn `Qwen/Qwen2.5-7B-Instruct`; vẫn có thể dùng
`--model` để chọn model causal LM tương thích khác.

> Đây là reproduction study dùng modern instruction-tuned LLM. Nó không phải
> bản tái hiện chính xác PaLM-540B của paper gốc, vì model, hạ tầng và trạng thái
> Wikipedia đều khác.

## ✅ Trạng thái hiện tại

Repository đã hoàn tất **Sprint 0 đến Sprint 6**. HotpotQA và FEVER đều hỗ trợ
đủ 7 phương pháp trong paper. ALFWorld, WebShop, sprint báo cáo nghiên cứu và
interactive app chưa bắt đầu.

| Sprint | Nội dung | Trạng thái |
|---:|---|---|
| 0 | Khung repository, config, logging, CLI | ✅ Hoàn tất |
| 1 | HotpotQA, metrics, schema và experiment runner | ✅ Hoàn tất |
| 2 | Hugging Face LLM, Standard và CoT | ✅ Hoàn tất |
| 3 | Wikipedia environment và Act-only | ✅ Hoàn tất |
| 4 | ReAct, live trajectory và lưu trajectory | ✅ Hoàn tất |
| Mở rộng | CoT-SC voting và hai thứ tự fallback ReAct/CoT-SC | ✅ Hoàn tất |
| 5 | Benchmark 7 phương pháp HotpotQA | ✅ Hoàn tất |
| 6 | Triển khai FEVER bám sát paper | ✅ Hoàn tất |
| 7+ | Báo cáo nghiên cứu, UI và phần mở rộng | ⏸️ Chưa bắt đầu |

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

### Ghi chú HotpotQA khi chạy trên Colab

- `--num-samples` là tổng số example. `--batch-size` quyết định số example của
  mọi method dùng chung một GPU generation call; model chỉ load một lần cho cả run.
- Split hiện tại `hotpotqa/hotpot_qa`, `distractor`, `validation` có tối đa
  **7.405 examples**. Giá trị hợp lệ là `1-7405`; nhập lớn hơn sẽ dừng với lỗi
  validation.
- Nên dùng 5 mẫu để smoke test, 20-50 mẫu để kiểm tra prompt/failure, và chỉ
  chạy 100/500+ sau khi đã xác nhận runtime cùng độ ổn định của MediaWiki.
- Không thêm `--quiet` trên Colab. Mỗi example hiện prediction, gold, Answer
  EM/F1, termination reason và operational metrics dạng gọn; toàn bộ 12 official
  HotpotQA metrics vẫn hiện cuối run và được lưu vào `metrics.json`/`run.log`.
- Prediction rỗng được ghi rõ là `<EMPTY>`. Supporting-fact/Joint chi tiết chỉ
  hiện với `--log-level DEBUG` vì milestone hiện tại chưa sinh evidence.
- Trước khi xử lý từng example, log hiển thị progress, `example_id` và câu hỏi
  để sample đang chạy luôn được nhận biết ngay trên Colab.
- `metrics.json` chứa answer `EM/F1/precision/recall`, supporting-fact
  `SP EM/F1/precision/recall`, joint `EM/F1/precision/recall`, evidence coverage,
  runtime, steps/tools trung bình và termination reasons.
- Công thức bám theo
  [official HotpotQA evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
- Agent hiện tại trả lời answer nhưng chưa xuất cặp evidence HotpotQA
  `(title, sentence_id)`. Evaluator chính thức sẽ chấm supporting-fact prediction
  rỗng (SP/joint thường bằng 0), báo rõ evidence coverage và không tự bịa evidence.
- Qwen2.5-7B cần nhiều bộ nhớ hơn đáng kể so với bản 3B trước đây. Nên chọn GPU
  Colab loại L4/A100. T4 có thể phải offload một phần model sang RAM nhờ
  `device_map=auto`, vì vậy tốc độ sẽ chậm hơn nhiều.

Chương trình tự chọn CUDA nếu PyTorch nhìn thấy GPU; nếu không có CUDA thì tự
chuyển sang CPU. `bitsandbytes` không bắt buộc. Có thể khai báo `HF_TOKEN` trong
Colab để tăng giới hạn tải từ Hugging Face Hub.

## ⌨️ Các lệnh chính

### HotpotQA

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
    --seed 42 \
    --batch-size 4
```

Chạy CoT-SC standalone theo default paper:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method cot-sc \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --batch-size 4 \
    --cot-sc-samples 21 \
    --cot-sc-temperature 0.7
```

Chạy CoT:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method cot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --batch-size 4
```

Chạy Act-only:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method act \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --batch-size 4 \
    --show-trajectories
```

Chạy ReAct:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method react \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --batch-size 4 \
    --show-trajectories
```

### Hai phương pháp hybrid trong paper

Chạy **ReAct → CoT-SC**. ReAct chạy trước; chỉ fallback sang CoT-SC khi ReAct
không tạo được `Finish` tự nhiên trong step budget:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method react-cot-sc \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --max-agent-steps 7 \
    --batch-size 2 \
    --cot-sc-samples 21 \
    --cot-sc-temperature 0.7
```

Chạy **CoT-SC → ReAct**. CoT-SC luôn chạy trước; fallback sang ReAct khi đáp án
đã normalize có số phiếu cao nhất xuất hiện ít hơn `n/2` lần:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method cot-sc-react \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --max-agent-steps 7 \
    --batch-size 2 \
    --cot-sc-samples 21 \
    --cot-sc-temperature 0.7
```

Hai lệnh trên cố ý giữ default của paper. Khi smoke test pipeline, có thể tạm
dùng `--cot-sc-samples 3`; benchmark kiểu paper dùng 21. Không nên bắt đầu ngay
với 500 examples: CoT-SC → ReAct cần ít nhất 21 lượt generation mỗi example
(10.500 lượt cho 500 examples), còn ReAct → CoT-SC chỉ chịu chi phí đó khi
fallback. Chỉ thêm `--show-trajectories` cho run nhỏ vì nó in mọi CoT sample và
mọi bước ReAct.

Trên A100, nên bắt đầu bằng `--batch-size 2`; thử 3 hoặc 4 nếu GPU vẫn còn đủ bộ nhớ.
Standard và CoT batch các prompt một lượt. CoT-SC batch các câu hỏi với nhau ở
từng vòng trong 21 vòng sampling. Act/ReAct batch những câu còn active tại cùng
step nhưng mỗi câu vẫn có Wikipedia state và trajectory riêng. Prediction vẫn
được flush theo thứ tự dataset sau mỗi batch, nên nếu runtime ngắt thì mất tối
đa batch đang xử lý. Batch size thay đổi lịch generation nên output sampling có
thể khác dù dùng cùng seed; phải ghi nhận batch size khi so sánh các run.

`--model` là tùy chọn; nếu bỏ qua, CLI tự dùng
`Qwen/Qwen2.5-7B-Instruct`. Standard, CoT, CoT-SC, Act-only, ReAct và hai hybrid nhận
prompt material 6-shot tương ứng trong Appendix C.1; example cần chấm vẫn chỉ
đưa câu hỏi vào model, không đưa supporting paragraphs.

Các tham số hữu ích:

| Tham số | Ý nghĩa |
|---|---|
| `--device auto` | Tự chọn CUDA, MPS hoặc CPU |
| `--num-samples 5` | Số câu hỏi được chạy |
| `--seed 42` | Cố định cách lấy mẫu và random seed |
| `--batch-size 2` | Số example mọi method chạy chung một GPU batch; A100 có thể thử 2, 3 hoặc 4 |
| `--max-agent-steps 7` | Số bước tối đa của Act/ReAct |
| `--max-new-tokens 256` | Số token sinh tối đa trong một lượt |
| `--cot-sc-samples 21` | Số CoT samples dùng để majority vote |
| `--cot-sc-temperature 0.7` | Sampling temperature riêng của CoT-SC |
| `--react-best-effort-finalization` | Ép ReAct dùng lượt cuối/recovery để `Finish` (không phải mặc định paper) |
| `--show-trajectories` | In từng Thought/Action/Observation |
| `--log-level INFO` | Mức chi tiết của log |

Act/ReAct coi `Search` là thao tác mở bài theo entity/title, không phải web
search. `Lookup` đọc chi tiết trong bài hiện tại. `Search` giống hệt lần hai sẽ
bị bỏ qua kèm hướng dẫn recovery; lần ba được ghi nhận là `action_loop`. ReAct
mặc định dùng policy đúng paper: có thể dùng đủ bảy bước và trả prediction rỗng
nếu không sinh `Finish`; đây cũng là điều kích hoạt fallback ReAct → CoT-SC.
ReAct standalone và ReAct trong cả hai hybrid dùng cùng policy để so sánh công
bằng. Cờ tùy chọn `--react-best-effort-finalization` ép một lượt cuối/recovery
để `Finish`, nhưng đây là override thử nghiệm và không nên dùng khi đối chiếu
bảng paper. Act-only vẫn giữ bước cuối best-effort.

### FEVER: đủ 7 phương pháp

Các lệnh dưới đây chạy 500 claims theo setting paper. Nên đổi thành
`--num-samples 3` để smoke test trước. FEVER dự đoán `SUPPORTS`, `REFUTES` hoặc
`NOT ENOUGH INFO`; ReAct dùng tối đa 5 bước, còn CoT-SC dùng 21 samples ở
temperature 0.7.

```bash
# 1. Standard
python main.py benchmark --task fever --method standard --model Qwen/Qwen2.5-7B-Instruct --num-samples 500 --seed 42 --batch-size 32

# 2. Chain-of-Thought
python main.py benchmark --task fever --method cot --model Qwen/Qwen2.5-7B-Instruct --num-samples 500 --seed 42 --batch-size 16

# 3. CoT với self-consistency
python main.py benchmark --task fever --method cot-sc --model Qwen/Qwen2.5-7B-Instruct --num-samples 500 --seed 42 --batch-size 16 --cot-sc-samples 21 --cot-sc-temperature 0.7

# 4. Act-only
python main.py benchmark --task fever --method act --model Qwen/Qwen2.5-7B-Instruct --num-samples 500 --seed 42 --batch-size 16 --max-agent-steps 5

# 5. ReAct
python main.py benchmark --task fever --method react --model Qwen/Qwen2.5-7B-Instruct --num-samples 500 --seed 42 --batch-size 8 --max-agent-steps 5

# 6. CoT-SC, fallback sang ReAct khi confidence thấp
python main.py benchmark --task fever --method cot-sc-react --model Qwen/Qwen2.5-7B-Instruct --num-samples 500 --seed 42 --batch-size 8 --max-agent-steps 5 --cot-sc-samples 21 --cot-sc-temperature 0.7

# 7. ReAct, fallback sang CoT-SC khi thất bại
python main.py benchmark --task fever --method react-cot-sc --model Qwen/Qwen2.5-7B-Instruct --num-samples 500 --seed 42 --batch-size 8 --max-agent-steps 5 --cot-sc-samples 21 --cot-sc-temperature 0.7
```

<a id="cách-hoạt-động"></a>

## 🧩 Bảy phương pháp CLI hoạt động như thế nào?

| Phương pháp | Có reasoning rõ ràng? | Dùng Wikipedia? | Luồng xử lý |
|---|---:|---:|---|
| Standard | Không | Không | Câu hỏi → câu trả lời |
| CoT | Có | Không | Câu hỏi → suy luận → câu trả lời |
| CoT-SC | Có | Không | 21 CoT samples → normalize → majority vote |
| Act-only | Không | Có | Action ↔ Observation → Finish |
| ReAct | Có | Có | Thought → Action → Observation → Finish |
| ReAct → CoT-SC | Có | Ở nhánh ReAct | ReAct; nếu fail, 21 CoT samples → vote |
| CoT-SC → ReAct | Có | Khi fallback | 21 CoT samples → vote; confidence thấp → ReAct |

Giải thích ngắn:

- **Standard** yêu cầu model trả lời trực tiếp từ kiến thức đã học.
- **CoT** yêu cầu model trình bày reasoning trước khi đưa ra đáp án.
- **Act-only** không lưu Thought; model chỉ chọn hành động Wikipedia.
- **ReAct** kết hợp reasoning và hành động. Observation luôn do environment
  tạo ra, không tin Observation do model tự viết.
- **ReAct → CoT-SC** giữ đáp án ReAct khi có `Finish` tự nhiên; nếu ReAct fail
  thì sample nhiều CoT và majority vote.
- **CoT-SC → ReAct** majority vote trước; chỉ gọi Wikipedia/ReAct khi vote cao
  nhất không đạt ngưỡng `n/2` của paper.

Core agent được viết trực tiếp bằng Python, không dùng LangChain, LangGraph,
CrewAI hay AutoGen. Nhờ vậy có thể đọc rõ parser, state, vòng lặp và nguyên nhân
dừng của từng example.

## 🏗️ Luồng chạy tổng thể

```text
main.py
  → CLI đọc tham số và config
  → HotpotQA hoặc FEVER loader lấy mẫu theo seed
  → HuggingFaceProvider load tokenizer/model
  → agent Standard / CoT / Act / ReAct / hybrid đã chọn
       └─ nhánh ReAct → WikipediaEnvironment → MediaWiki API
  → evaluator theo task: HotpotQA metrics hoặc FEVER Accuracy
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
│   ├── agents/                     # Base agents, CoT-SC voting, hybrid policies
│   ├── datasets/                   # BenchmarkExample và loader HotpotQA/FEVER
│   ├── evaluation/                 # Full official metrics và schema kết quả
│   ├── experiments/                # Runner và ghi artifact
│   ├── llm/                        # Interface LLM và Hugging Face provider
│   ├── prompts/                    # Prompt registry cho HotpotQA/FEVER
│   ├── tools/                      # Wikipedia client/environment
│   ├── cli.py                      # Khai báo command và kết nối component
│   ├── config.py                   # Đọc, kiểm tra YAML và biến môi trường
│   └── logging_utils.py            # Live stdout UTF-8 và run.log
└── tests/                           # 146 unit/integration-style tests
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
outputs/<task>/<method>/<UTC timestamp>/
├── config.json
├── metrics.json
├── predictions.jsonl
├── trajectories.jsonl
└── run.log
```

| File | Chức năng |
|---|---|
| `config.json` | Lưu model, dataset, method, seed, generation config và device |
| `predictions.jsonl` | Mỗi dòng là kết quả một example; flush theo thứ tự sau mỗi batch |
| `trajectories.jsonl` | Mỗi dòng là trajectory có phase Act/ReAct/hybrid |
| `metrics.json` | Full official HotpotQA metrics, runtime, steps/tool calls và lý do dừng |
| `run.log` | Bản log được lưu song song với stdout |

`trajectories.jsonl` của hybrid ghi phase `cot_sc`/`react`, model output,
Thought, Action và Observation. `predictions.jsonl` còn ghi vote count,
confidence, nhánh được chọn và có fallback hay không. `--show-trajectories` in
các thông tin này để debug trên terminal hoặc Colab.

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

146 tests hiện tại kiểm tra:

- CLI, cấu hình và `doctor`;
- HotpotQA/FEVER loading, validation và sampling theo seed;
- answer EM/F1/precision/recall, supporting-fact và joint metrics chính thức;
- 6-shot prompt pack của paper, nhãn đánh số và Standard/CoT parser;
- Search/Lookup/Finish, trang thiếu hoặc mơ hồ;
- Lookup nhiều lần, loop detection và max-step;
- Act-only/ReAct, parsing recovery và trajectory history;
- CoT-SC voting/threshold cùng cả hai thứ tự hybrid fallback;
- flush `predictions.jsonl` và `trajectories.jsonl` theo từng example.

Unit tests không tải model và không gọi Wikipedia thật. Chúng sử dụng scripted
LLM, dữ liệu inject và fake Wikipedia client nên chạy nhanh, ổn định.

## ⚙️ Dependency chính

- `torch`, `transformers`, `accelerate`: chạy model Hugging Face và tự đặt
  device;
- `datasets`: tải/cache HotpotQA và FEVER source đã pin;
- `requests`, `tenacity`: gọi MediaWiki có timeout và retry;
- `PyYAML`, `python-dotenv`: cấu hình;
- `pytest`: chạy test repository.

Model được load lazy, chỉ tải khi chạy benchmark. CUDA ưu tiên BF16 nếu GPU hỗ
trợ, nếu không dùng FP16; CPU dùng FP32.

## ⚠️ Giới hạn

- HotpotQA đã có đủ 7 method runs; kết quả FEVER 500 claims vẫn đang chờ chạy.
- Không thể kỳ vọng tái hiện chính xác số liệu PaLM-540B của paper gốc.
- Wikipedia và kết quả search có thể thay đổi theo thời gian.
- CoT-SC kiểu paper sinh 21 lượt cho mỗi example, nên hybrid tốn thời gian và
  GPU quota hơn ReAct thuần đáng kể.
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
