# RAG Development Flow

## 1. Chuẩn bị dữ liệu đánh giá

- Lưu corpus gốc trong `data/raw/` và không chỉnh sửa trực tiếp.
- Lưu golden dataset trong `data/evaluation/golden_recipes_100_en.jsonl`.
- Golden dataset hiện có 100 mẫu hỏi đáp để dùng làm chuẩn đánh giá các layer RAG.

**Trạng thái:** Hoàn thành.

## 2. Kiểm tra và chuẩn hóa golden dataset

- Chuẩn hóa `gold_sources` theo đường dẫn thật trong `data/raw/`.
- Kiểm tra JSON hợp lệ, đủ trường bắt buộc và `id` không trùng.
- Kiểm tra toàn bộ tài liệu nguồn tồn tại.
- Sử dụng golden dataset tiếng Anh tại `data/evaluation/golden_recipes_100_en.jsonl`, gồm 100 records hợp lệ.

**Trạng thái:** Hoàn thành.

## 3. Thiết lập environment

- Sử dụng Python 3.12 và virtual environment `.venv`.
- Quản lý project và dependency bằng `pyproject.toml`.
- Cài project ở editable mode để code trong `src/` có thể import khi phát triển.
- Cài `pytest` để test độc lập từng layer.
- Thêm `.venv`, `.env` và Python cache vào `.gitignore`.
- Tạo template song song cho từng layer trong `src/`, `scripts/` và `tests/`.
- Đã verify `recipe-rag` và `pytest` chạy trong đúng `.venv`.

**Trạng thái:** Hoàn thành.

## 4. Ingestion

- Load toàn bộ Markdown thành document gồm `document_id`, `content` và `metadata`.
- Dùng script để nhập tay một file và quan sát output.
- Full-corpus test đã load và kiểm tra đủ 85 documents.

**Trạng thái:** Hoàn thành.

## 5. Cleaning

- Chuẩn hóa newline, trailing whitespace và dòng trống dư.
- Giữ nguyên nội dung, cấu trúc Markdown và metadata.
- Full-corpus test đã clean thành công 85 documents.
- Lưu 85 cleaned documents vào `data/processed/cleaned_documents.jsonl`.
- Verify toàn bộ 85 dòng parse được dưới dạng JSON.

**Trạng thái:** Hoàn thành.

## 6. Chunking

- Chia mỗi parent recipe thành child chunks theo Markdown section.
- Mỗi chunk giữ title, section, source và `parent_document_id`.
- Loại section `based on` khỏi dữ liệu dùng cho retrieval.
- Full-corpus test tạo 272 chunk từ 85 parent documents.
- Lưu và verify `data/processed/chunks.jsonl` gồm 272 dòng JSON hợp lệ.

**Trạng thái:** Hoàn thành.

## 7. Embedding

- Tạo một embedding interface dùng chung cho BGE, E5 và MiniLM.
- Chọn model bằng `active_model` trong `configs/embedding.yaml`.
- Mỗi model profile lưu vectors riêng tại `data/processed/embeddings/<profile>.jsonl`.
- Hoàn thiện manual inspector, full-corpus test và build script.
- Đã tạo 272 vectors BGE 384 chiều tại `data/processed/embeddings/bge_small.jsonl`.

**Trạng thái:** Hoàn thành.

## 8. Vector store

- Dùng ChromaDB persistent collection với cosine distance.
- Collection lưu vector, content và metadata của từng chunk.
- Mỗi embedding profile có storage riêng tại `data/vector_store/<profile>/`.
- Full-corpus test kiểm tra build, reload, search và metadata filtering trên 272 chunks.
- Chroma index BGE thật đã persist đủ 272 records.

**Trạng thái:** Hoàn thành.

## 9. Retrieval

- Embed query theo active profile, kiểm tra model/dimension rồi search Chroma top-k.
- Script nhận một câu hỏi và in rank, score, source, section cùng content để kiểm tra thủ công.
- Test batch 100 golden questions; metrics tính trên 88 câu answerable.
- Baseline BGE: Hit@1 `0.5341`, Hit@3 `0.8068`, Hit@5 `0.8864`, MRR@5 `0.6716`, source Recall@5 `0.8580`.
- Retrieval test in rõ embedding profile và model đang được đánh giá.
- Mỗi profile tạo error report tại `data/processed/evaluation/retrieval_errors_<profile>.csv`.
- Report giữ top 5 của các câu `miss@1` và `miss@5` để lọc, so sánh và debug bằng Excel.
- Thêm BM25 in-memory đọc trực tiếp 272 chunks, không cần embedding hoặc vector store.
- Thêm hybrid retriever gộp dense và BM25 bằng weighted Reciprocal Rank Fusion (RRF); hỗ trợ chỉnh `dense_weight` và `bm25_weight`.
- Chọn `dense`, `bm25` hoặc `hybrid` cùng candidate sizes và fusion weights trong `configs/retrieval.yaml`.
- Retrieval inspector và golden benchmark cùng đọc retrieval config, tránh sửa strategy trực tiếp trong code.
- `test_retrieval_strategies.py` kiểm tra độc lập rule của dense, BM25 và hybrid, không phụ thuộc strategy đang active trong config.
- Baseline hybrid BGE với weights `1.0:1.0`: Hit@1 `0.5114`, Hit@3 `0.7955`, Hit@5 `0.8750`, MRR@5 `0.6538`, source Recall@5 `0.8371`.
- Tuned hybrid BGE với weights `1.5:0.5`: Hit@1 `0.5682`, Hit@3 `0.8182`, Hit@5 `0.8864`, MRR@5 `0.6939`, source Recall@5 `0.8390`.

**Trạng thái:** Hoàn thành.

## Các bước tiếp theo

10. Xây dựng generation.
11. Đánh giá chất lượng RAG.
12. Tạo demo deploy.
13. Thiết lập CI/CD.
