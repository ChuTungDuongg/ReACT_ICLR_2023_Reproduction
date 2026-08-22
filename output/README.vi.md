# Tài liệu của project

> 🌐 **Ngôn ngữ:** [English](README.md) | Tiếng Việt

`output/` chứa tài liệu ổn định dành cho người đọc và có thể commit/chia sẻ.
Thư mục này khác `outputs/`, nơi chứa benchmark runtime bị Git ignore.

```text
output/
└── pdf/
    └── react-reproduction-roadmap.pdf
```

## Roadmap PDF ghi lại những gì?

- mục tiêu và trạng thái của từng sprint;
- architecture và trách nhiệm của từng folder/file;
- requirements Python, Colab và hardware;
- acceptance criteria và checkpoint của Sprint 0-4;
- cấu trúc artifact và lệnh kiểm tra;
- smoke benchmark thật cùng kết quả thực tế;
- phần roadmap còn lại được đánh dấu rõ là chưa triển khai.

Roadmap hiện phản ánh trạng thái **hoàn tất đến Sprint 4**. Sprint 5, FEVER,
ALFWorld, WebShop và interactive app vẫn chưa được thực hiện.

## Không đặt ở đây

- prediction hoặc trajectory runtime;
- model weights, cache, dataset;
- credential hoặc token;
- file tạm dùng khi render tài liệu.

Các nội dung trên phải nằm ở thư mục phù hợp và tuân theo `.gitignore`.
