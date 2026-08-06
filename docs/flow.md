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

- Hỗ trợ `dense`, `bm25` và `hybrid`; strategy, candidate sizes và weights được chọn trong `configs/retrieval.yaml`.
- Dense search dùng query embedding và Chroma; BM25 đọc trực tiếp chunks; hybrid gộp hai bảng xếp hạng bằng weighted RRF.
- Reranker nhận top 20 từ hybrid, dùng cross-encoder đọc từng cặp `question + chunk`, sắp xếp lại rồi trả final top 5.
- Cross-encoder chỉ inference bằng pretrained weights, không train trên golden dataset và không tạo index mới.
- Reranker chỉ có thể cứu chunk đã xuất hiện trong candidate top 20; đổi lại chất lượng cải thiện đáng kể nhưng latency CPU tăng.
- Strategy unit test kiểm tra rule độc lập; golden test đánh giá 100 câu và xuất CSV top 5 false cases để debug.
- Kết quả cuối: Hit@1 `0.7045`, Hit@3 `0.8977`, Hit@5 `0.9205`, MRR@5 `0.7973`, Source Recall@5 `0.9015`.

**Trạng thái:** Hoàn thành.

## 10. Generation

- Nhận question cùng top 5 contexts sau rerank và gọi model qua OpenAI-compatible API.
- Prompt chỉ cho phép dùng supplied context, bắt buộc citation `[n]`, từ chối khi thiếu dữ liệu và bỏ qua instruction nằm trong context.
- Guardrails giới hạn question/context, không gọi API khi context rỗng và chuyển sang refusal nếu output không có citation hợp lệ.
- API key, base URL và model nằm trong `.env`; behavior và limits nằm trong `configs/generation.yaml`.
- Ba unit tests dùng fake client đã pass; full flow thật trả đúng Garlic Bread ingredients và citation đúng source/section.

**Trạng thái:** Hoàn thành baseline.

## Các bước tiếp theo

11. Đánh giá chất lượng RAG.
12. Tạo demo deploy.
13. Thiết lập CI/CD.
