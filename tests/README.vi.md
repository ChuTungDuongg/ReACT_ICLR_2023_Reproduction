# Kiểm thử

> 🌐 **Ngôn ngữ:** [English](README.md) | Tiếng Việt

Repository hiện có **68 tests**. Unit tests không tải LLM và không gọi
Wikipedia thật; chúng dùng scripted LLM, dữ liệu inject và fake Wikipedia
client để kiểm tra đúng control flow với tốc độ nhanh, ổn định.

Chạy tất cả test từ root:

```bash
python -m pytest -q
```

Chạy riêng một nhóm khi debug:

```bash
python -m pytest tests/test_react_agent.py -q
python -m pytest tests/test_wikipedia.py -q
python -m pytest tests/test_experiment_serialization.py -q
```

## Từng file kiểm tra gì?

| File | Nội dung kiểm tra |
|---|---|
| `conftest.py` | Thêm `src/` vào import path cho pytest |
| `test_smoke.py` | CLI help, config và `doctor` |
| `test_hotpotqa.py` | Sampling theo seed, metadata, loader args và validation |
| `test_hotpotqa_prompts.py` | 6 ví dụ Appendix C.1 cho từng method, target question-only và history đánh số |
| `test_metrics.py` | Official answer, supporting-fact và joint HotpotQA metrics |
| `test_parsing.py` | Parse đáp án Standard/CoT và reasoning |
| `test_standard_cot_agents.py` | Prompt, result và trajectory của Standard/CoT |
| `test_hybrid_agents.py` | CoT-SC voting, threshold paper và hai thứ tự fallback |
| `test_action_parsing.py` | Search/Lookup/Finish, Thought/Action và format drift |
| `test_wikipedia.py` | Search, Lookup tiếp theo, missing, ambiguity, loop, max-step |
| `test_act_agent.py` | Vòng lặp Act-only và phục hồi parsing error |
| `test_react_agent.py` | Thought/Action/Observation, history và termination |
| `test_experiment_serialization.py` | Config, full metrics, terminal log, prediction/trajectory flush |

## Fake và integration-style khác gì benchmark thật?

- **Unit test** thay model/network bằng fake để kiểm tra logic xác định.
- **Integration-style test** nối nhiều component thật nhưng vẫn dùng fake ở
  biên model/network.
- **Smoke benchmark thật** mới tải Hugging Face model và gọi MediaWiki.

Không được coi output của fake/mock là benchmark result.

## Quy tắc thêm test

- Test phải deterministic và không phụ thuộc kết nối internet.
- Mỗi bug parser/loop cần một regression test.
- File tạm chỉ được tạo dưới `outputs/`.
- Trước khi xóa output test, phải kiểm tra resolved path đúng nằm dưới
  `outputs/`.
- Test trajectory cần kiểm tra cả raw output, Thought, Action và Observation.
