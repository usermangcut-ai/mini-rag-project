```
RAG/
├── data/
│   ├── raw/                 # Corpus gốc, không chỉnh sửa
│   ├── processed/           # Dữ liệu sau cleaning
│   └── evaluation/          # Golden dataset
├── src/
│   ├── ingestion/
│   ├── cleaning/
│   ├── chunking/
│   ├── embedding/
│   ├── vector_store/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
├── tests/                   # Test độc lập cho từng layer
├── configs/                 # Model, chunk size, top-k...
├── scripts/                 # Các lệnh chạy pipeline
├── app/                     # Demo deploy
├── .github/workflows/       # CI/CD, tạo sau
├── .env.example             # Tên biến môi trường, không chứa secret
├── .gitignore
├── pyproject.toml
└── README.md
```
