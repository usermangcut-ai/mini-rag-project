# Mini RAG Project

Đây là một dự án RAG nhỏ được xây dựng trên corpus công thức nấu ăn. Mục tiêu chính không chỉ là tạo ra một pipeline chạy được, mà còn là hiểu, quan sát và debug từng layer độc lập.

## Mục tiêu

- Xây dựng pipeline RAG theo từng layer rõ ràng.
- Cho phép thay đổi model và tham số bằng config.
- Test tự động từng layer bằng `pytest`.
- Quan sát output thực tế bằng các script cần thiết.
- Đánh giá retrieval và generation bằng golden dataset.
- So sánh các embedding model bằng cùng một bộ dữ liệu đánh giá.

## Pipeline

```text
Raw Markdown
    → Ingestion
    → Cleaning
    → Chunking
    → Embedding
    → Vector Store
    → Retrieval
    → Generation
    → Evaluation
```

Mỗi layer trong `src/` chịu trách nhiệm cho một phần của pipeline. Các artifact trung gian được tạo bằng script build, còn `tests/` dùng để kiểm tra toàn bộ corpus và phát hiện lỗi ở từng tầng.

## Hướng đánh giá

Golden dataset được dùng làm chuẩn để đo chất lượng retrieval và generation. Khi thay đổi embedding model, chunking strategy hoặc tham số tìm kiếm, kết quả benchmark sẽ được so sánh trên cùng golden dataset thay vì đánh giá cảm tính bằng một vài câu hỏi.

Các chỉ số retrieval chính gồm:

- Hit@k
- MRR@k
- Source Recall@k

Ngoài test tự động, kết quả retrieval vẫn cần được quan sát thủ công để phát hiện lỗi từ dữ liệu, chunking, metadata hoặc nhãn trong golden dataset.
