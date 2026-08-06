# Retrieval Layer

## Source logic

- Embed the user question with the active model profile.
- Validate that query model and dimension match the Chroma index.
- Search Chroma with cosine distance and return ranked top-k chunks.

## Script logic

- `scripts/inspect_retrieval.py` accepts one question from PowerShell.
- It prints rank, score, source, section, and full chunk content for manual review.

```powershell
python .\scripts\inspect_retrieval.py
```

## Test logic

- `tests/retrieval/test_retriever.py` embeds all 100 golden questions in one batch.
- Retrieval metrics use answerable questions only; unanswerable questions are evaluated later at generation/refusal level.
- It reports Hit@1, Hit@3, Hit@5, MRR@5, and source Recall@5.

```powershell
python -m pytest -s .\tests\retrieval\test_retriever.py
```
