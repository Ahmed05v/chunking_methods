# Arabic pilot chunking result

The 11 Arabic article collections total **407,631 `o200k_base` tokens**:
3 technology, 5 legal, and 3 social collections. All eight chunking methods
completed on **11/11 documents**. Every stored method/stage covers the source
text, allowing the intentional overlap of LangChain's default splitter.

| Method | Raw chunks | Mean tokens | Raw chunks ≤1,100 tokens |
|---|---:|---:|---:|
| Page | 230 | 1,772.3 | 3.9% |
| Stanza Arabic sentence | 570 | 715.4 | 82.8% |
| LangChain recursive default | 435 | 943.1 | 71.7% |
| LangChain recursive 1,100 | 536 | 761.3 | 99.6% |
| Project recursive 1,100 | 469 | 869.2 | 100% |
| Project recursive 600 | 863 | 472.6 | 100% |
| Semantic, local Qwen3-Embedding-0.6B | 473 | 862.5 | 72.1% |
| Heading regex approximation | 1,513 | 269.4 | 98.1% |

The four postprocessed methods (page, sentence, semantic, and heading regex)
have additional results after splitting oversized chunks and merging small
ones. See [`chunk_size_summary.csv`](chunk_size_summary.csv) for all stages.
See [`PAPER_COMPARISON.md`](PAPER_COMPARISON.md) for the side-by-side paper
comparison. Arabic ICC and DCC were measured with local Qwen embeddings and
are labeled as approximations in [`intrinsic_vs_paper.csv`](intrinsic_vs_paper.csv).
`chunks/raw/chunks.parquet` stores the raw chunks; `chunks/no_oversizing/` and
`chunks/small_merged/` store those postprocessed methods. Numbers above are
basic chunk-size/coverage checks. Arabic reference completeness and RAG
accuracy remain unmeasured.

The source collections are assembled from attributed Arabic Wikipedia text.
Their pages and paragraph boundaries are synthetic. See
[`data/arabic_pilot/README.md`](../../data/arabic_pilot/README.md) and
[`manifest.json`](../../data/arabic_pilot/manifest.json) for source and
licensing details. The local heading rule is not GPT-5. The English Maverick
coreference model and the paper's question/answer set do not provide valid
Arabic reference-completeness or RAG scores.

To regenerate the size report after chunking:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_arabic_pilot.py
```
