# Cấu hình

> 🌐 **Ngôn ngữ:** [English](README.md) | Tiếng Việt

Thư mục này chứa cấu hình mặc định có version control. Hiện tại file chính là
`default.yaml`.

## `default.yaml` chứa gì?

```yaml
project_name: react-paper-reproduction
output_dir: outputs
log_level: INFO

datasets:
  hotpotqa:
    dataset_name: hotpotqa/hotpot_qa
    subset: distractor
    split: validation

benchmark:
  num_samples: 10
  seed: 42
  max_agent_steps: 7

generation:
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 256
```

## Thứ tự ưu tiên

1. Giá trị truyền bằng CLI dùng cho run hiện tại.
2. Biến môi trường được hỗ trợ như `REACT_OUTPUT_DIR`, `REACT_LOG_LEVEL`.
3. Giá trị trong `configs/default.yaml`.
4. Default trong dataclass nếu YAML không khai báo.

`src/react_reproduction/config.py` đọc file YAML, kiểm tra kiểu/range rồi resolve
đường dẫn tương đối từ root repository. Giá trị không hợp lệ sẽ dừng sớm với
thông báo rõ ràng trước khi tải dataset hoặc model.

## Không được lưu ở đây

- Hugging Face token hoặc credential;
- model weight, dataset đã tải;
- output benchmark;
- đường dẫn tuyệt đối chỉ đúng trên một máy.

Token local nên đặt trong `.env`; file `.env` đã được Git ignore. Không commit
secret vào `default.yaml`.
