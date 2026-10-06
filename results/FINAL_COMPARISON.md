# Final local results versus the paper

Source: [Adaptive Chunking, LREC 2026](https://aclanthology.org/2026.lrec-1.903.pdf), Tables 1–7. Local files: `results/`.

**RC** = reference completeness; **ICC** = cohesion within chunks; **DCC** = agreement with nearby document context; **BI** = intact structural blocks; **SC** = chunks within the 100–1,100 token size range. All scores are percentages. A cell is **local mean ± standard deviation / paper mean ± standard deviation**.

| Method | RC (local / paper) | ICC (local / paper) | DCC (local / paper) | BI (local / paper) | SC (local / paper) | Overall local / paper | Δ points |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM regex (local heuristic approximation) | 97.5±3.9 / 98.0±2.9 | 71.7±5.1 / 70.9±5.1 | 82.4±5.5 / 82.4±5.5 | 98.8±1.4 / 98.1±2.5 | 99.6±1.1 / 99.6±1.3 | **90.01 / 89.80** | +0.21 |
| Our recursive (1100), post-processed | 99.0±1.8 / 99.0±1.8 | 66.6±3.8 / 66.6±3.8 | 89.8±2.5 / 89.7±2.5 | 98.1±1.9 / 98.1±1.9 | 100.0±0.1 / 100.0±0.1 | **90.68 / 90.68** | +0.00 |
| Our recursive (600), post-processed | 97.2±2.9 / 97.2±2.9 | 69.6±4.1 / 69.6±4.1 | 84.7±3.0 / 84.7±3.1 | 94.8±3.6 / 94.8±3.6 | 100.0±0.0 / 100.0±0.0 | **89.25 / 89.24** | +0.01 |
| Page, post-processed | 97.2±3.5 / 97.2±3.5 | 69.2±3.6 / 69.2±3.6 | 86.4±4.6 / 86.4±4.6 | 99.9±0.3 / 99.9±0.3 | 99.9±0.4 / 99.9±0.4 | **90.52 / 90.52** | +0.00 |
| LangChain recursive (1100), raw | 98.4±3.1 / 98.4±3.1 | 64.8±3.4 / 64.7±3.4 | 86.9±2.9 / 86.8±2.8 | 98.6±1.4 / 98.6±1.4 | 93.3±7.2 / 93.3±7.2 | **88.40 / 88.37** | +0.03 |
| LangChain recursive (default), raw | 96.1±4.3 / 96.1±4.3 | 65.7±3.6 / 65.6±3.6 | 88.8±2.4 / 88.8±2.4 | 95.0±2.8 / 95.0±2.8 | 97.7±6.3 / 97.7±6.3 | **88.64 / 88.62** | +0.02 |
| Page, raw | 97.1±3.5 / 97.1±3.5 | 69.4±3.4 / 69.3±3.4 | 86.2±4.9 / 86.1±5.0 | 100.0±0.0 / 100.0±0.0 | 92.7±9.7 / 92.7±9.7 | **89.07 / 89.03** | +0.04 |
| Semantic (local Qwen approximation) | 97.0±3.9 / 97.5±3.1 | 69.3±4.1 / 69.3±4.1 | 76.3±7.2 / 76.3±7.3 | 91.3±4.0 / 91.3±4.0 | 48.1±17.5 / 48.1±17.6 | **76.38 / 76.49** | -0.11 |
| Sentence, raw | 87.0±8.9 / 86.3±11.1 | 78.7±2.6 / 78.4±2.7 | 72.5±5.8 / 72.5±5.8 | 62.2±10.0 / 61.9±10.3 | 64.4±24.0 / 67.2±23.1 | **72.97 / 73.26** | -0.29 |
| Adaptive selection (local) | 98.7±2.0 / 99.0±1.5 | 68.7±4.5 / 68.2±4.7 | 88.4±4.5 / 88.8±3.5 | 99.4±1.2 / 99.4±1.2 | 99.9±0.2 / 99.9±0.3 | **91.03 / 91.07** | -0.04 |

The LLM regex row is a local heading rule in our run; the paper used GPT-5. The semantic row uses our local Qwen chunker, then Jina v3 for **all 33 real ICC and DCC scores**. RC has 31 available English documents; other individual scores have between 30 and 33 valid documents. See `table3_comparison.csv` for each score's exact sample count. Adaptive selection uses our locally selected method per document, so its row can differ from the paper's selection.

## Chunk sizes (paper Table 2)

| Method | Local chunks / paper chunks | Mean tokens: local / paper |
|---|---:|---:|
| LLM regex (local heuristic approximation) | 2,270 / 2,279 | 520 / 518 |
| Our recursive (1100), post-processed | 1,345 / 1,345 | 878 / 878 |
| Our recursive (600), post-processed | 2,381 / 2,381 | 496 / 496 |
| Page, post-processed | 1,780 / 1,780 | 663 / 663 |
| LangChain recursive (1100), raw | 1,675 / 1,675 | 706 / 706 |
| LangChain recursive (default), raw | 1,557 / 1,557 | 773 / 773 |
| Page, raw | 1,765 / 1,765 | 669 / 669 |
| Semantic (local Qwen approximation) | 1,706 / 1,706 | 693 / 693 |
| Sentence, raw | 8,498 / 8,125 | 139 / 146 |
| Adaptive selection (local) | 1,835 / 1,631 | 643 / 724 |

## Corpus (paper Table 1)

| Domain | Documents: local / paper | Total tokens: local / paper | Mean pages: local / paper |
|---|---:|---:|---:|
| Tech | 9 / 9 | 47,315 / 47,313 | 12 / 12 |
| Legal | 16 / 16 | 494,313 / 494,320 | 46 / 46 |
| Social sciences | 8 / 8 | 638,898 / 638,896 | 114 / 114 |

## Size compliance before and after processing (paper Table 7)

| Method | Raw SC: local / paper | Final SC: local / paper |
|---|---:|---:|
| LLM regex (local heuristic approximation) | 58.1% / 58.3% | 99.6% / 99.6% |
| Our recursive (1100), post-processed | 100.0% / 100.0% | 100.0% / 100.0% |
| Our recursive (600), post-processed | 98.7% / 98.7% | 100.0% / 100.0% |
| Page, post-processed | 92.7% / 92.7% | 99.9% / 99.9% |
| LangChain recursive (1100), raw | 93.3% / 93.3% | 99.4% / 99.4% |
| LangChain recursive (default), raw | 97.7% / 97.7% | n/a |
| Semantic (local Qwen approximation) | 48.1% / 48.1% | 99.9% / 99.9% |
| Sentence, raw | 64.4% / 67.2% | 100.0% / 100.0% |

These local SC values average each document's size-compliance score. The paper's full Table 7 also reports mean quality scores at the intermediate oversized-split stage, which we have not fully rescored.

## Adaptive method choice (paper Table 4)

| Selected method | Local documents | Local share | Paper share |
|---|---:|---:|---:|
| page | 14 | 42.4% | 48% |
| our_recurs_1100 | 13 | 39.4% | 42% |
| our_recurs_600 | 1 | 3.0% | 3% |
| llm_regex | 5 | 15.2% | 6% |

## RAG evaluation (paper Table 5)

These are **different tests**. The paper used its GPT-4.1 generated questions, answer generation, and GPT-4.1 judging. Our 99 questions were created separately; TF-IDF retrieved chunks and word overlap gave a simple proxy. The proxy is **not** an estimate of GPT-4.1 answer correctness or retrieval completeness.

| Method | Our lexical retrieval hit* | Our answer-word overlap* | Paper retrieval completeness | Paper answer correctness | Paper mean |
|---|---:|---:|---:|---:|---:|
| best | 100.0% | 53.3% | 67.68% | 78.01% | 71.77% |
| langch_recurs_default | 100.0% | 55.0% | 58.08% | 70.11% | 62.07% |
| page | 100.0% | 46.1% | 59.09% | 73.33% | 63.80% |

*Local retrieval hit means the best TF-IDF match was above zero; answer-word overlap measures how many reference-answer words appear in that chunk. They are diagnostic proxies, not the paper's metrics. The paper reports 65 answered queries for adaptive selection and 49 for each baseline; our proxy did not generate answers.

## Models and tools used

| Task | Paper | Our run |
|---|---|---|
| Documents and parsing | 33 CLAIR documents with parsed text and structural boundaries | Same 33 parsed source documents |
| LLM regex chunks | GPT-5 selects a regex from a document sample | Fixed local heading-based regex; no LLM call |
| Semantic chunks | LangChain experimental semantic splitter with gradient thresholding | Same splitter type with local Qwen3-Embedding-0.6B; raw chunks used for Table 3 |
| Recursive and page chunks | Authors' recursive splitter, LangChain recursive splitter, page splitter | Repository implementations of those splitters |
| Sentence chunks | Stanza, five sentences per chunk | Stanza, five sentences per chunk |
| ICC and DCC scoring | Jina `jina-embeddings-v3` sentence embeddings | Local Jina `jina-embeddings-v3`; semantic ICC and DCC rerun across 33 documents |
| Reference links (RC) | Maverick `maverick-mes-ontonotes` entity/pronoun extraction | Precomputed Maverick mention files; 31 documents yield RC scores |
| Tokens, size, structural integrity | OpenAI `o200k_base` token encoding; parser block boundaries | `tiktoken` and the parsed block boundaries |
| RAG questions | GPT-4.1, three per document | 99 separately authored replacement questions, three per document |
| RAG retrieval | Qwen3-Embedding-4B dense search + BM25 + Snowflake reranking | Local TF-IDF retrieval proxy |
| RAG answers and scoring | GPT-4.1 generation and GPT-4.1/DeepEval judging | Answer-word overlap proxy; no generative model or LLM judge |

The paper's exact RAG Table 5 remains unreproduced because its original question set and GPT-4.1 evaluation were not used here. Table 6 local metric runtimes are partial after resumed runs and cannot be compared fairly with the paper's 36:11 total. `table7_postprocessing_effect.csv` contains the local raw-versus-postprocessed chunk-size data; the paper's Table 7 also reports scores at intermediate stages, which were not all recomputed locally.
