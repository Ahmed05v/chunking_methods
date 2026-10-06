# Arabic pilot versus the paper

Source: [Adaptive Chunking: Optimizing Chunking-Method Selection for RAG](https://aclanthology.org/2026.lrec-1.903.pdf), Tables 1–7. Arabic results: [`chunk_size_summary.csv`](chunk_size_summary.csv). This compares **11 assembled Arabic Wikipedia article collections** with **33 natural English/French CLAIR PDFs**. It is descriptive, not a controlled language-only experiment. All Arabic methods covered 11/11 documents.

## Corpus (paper Table 1)

| Domain | Arabic docs / paper docs | Arabic tokens / paper tokens | Arabic mean tokens per doc / paper | Arabic pages / paper mean pages |
|---|---:|---:|---:|---:|
| Technology | 3 / 9 | 14,530 / 47,313 | 4,843 / 5,257 | 3 synthetic / 12 real |
| Legal | 5 / 16 | 153,567 / 494,320 | 30,713 / 30,895 | 17.8 synthetic / 46 real |
| Social | 3 / 8 | 239,534 / 638,896 | 79,845 / 79,862 | 44 synthetic / 114 real |
| **Total** | **11 / 33** | **407,631 / 1,180,529** | **37,057 / 35,774** | — |

The Arabic token lengths approximate the domain means, but the pages are artificial paragraph groups of about 2,000 tokens. The natural-document size range and page layout of the paper were not matched.

## Chunk output and size compliance (paper Tables 2 and 3)

Arabic **SC** uses the paper's rule: the percentage of chunks with **100–1,100 tokens**, averaged over documents. The paper's values also average document scores. `†` rows use raw chunks; other rows use the final postprocessed chunks.

| Method | Arabic chunks / paper chunks | Mean tokens: Arabic / paper | SC: Arabic / paper |
|---|---:|---:|---:|
| Heading regex approximation / paper GPT-5 regex | 1,148 / 2,279 | 355 / 518 | 99.9% / 99.6% |
| Project recursive 1,100 | 469 / 1,345 | 869 / 878 | 99.8% / 100.0% |
| Project recursive 600 | 863 / 2,381 | 473 / 496 | 99.5% / 100.0% |
| Page, postprocessed | 512 / 1,780 | 796 / 663 | 99.5% / 99.9% |
| † LangChain recursive 1,100 | 536 / 1,675 | 761 / 706 | 93.4% / 93.3% |
| † LangChain recursive default | 435 / 1,557 | 943 / 773 | 68.7% / 97.7% |
| † Page, raw | 230 / 1,765 | 1,772 / 669 | 3.6% / 92.7% |
| † Semantic, local Qwen | 473 / 1,706 | 863 / 693 | 48.9% / 48.1% |
| † Stanza sentence | 570 / 8,125 | 715 / 146 | 79.5% / 67.2% |

The Arabic corpus is roughly one-third as large overall, so **chunk counts should not be interpreted as a win/loss**. The biggest SC difference is raw page chunking: the pilot's artificial pages average ~2,000 tokens by construction, while paper pages are much shorter. LangChain's default splitter uses character length; Arabic text produced many chunks over the 1,100-token evaluation limit. The sentence splitter uses five Arabic sentences per chunk, which produced much longer chunks than five sentences in the paper corpus. Similar semantic SC values do not establish similar semantic quality.

## Quality metrics (paper Table 3)

**BI proxy** below uses Arabic *paragraph boundaries*. The paper's BI uses Azure parser structural blocks, including tables and figures; these are different targets and the scores should not be directly compared. RC, ICC, and DCC were **not measured** for this Arabic pilot. Arabic RC requires an Arabic coreference model; the paper explicitly notes that its Maverick implementation supports English only. ICC/DCC require separately validated Arabic embedding evaluation. Consequently, an Arabic five-metric mean or adaptive winner cannot be calculated.

| Method | Arabic BI paragraph proxy | Paper BI | Paper RC | Paper ICC | Paper DCC | Paper 5-metric mean |
|---|---:|---:|---:|---:|---:|---:|
| Heading regex / GPT-5 regex | 98.8% | 98.1% | 98.0% | 70.9% | 82.4% | 89.80% |
| Project recursive 1,100 | 97.5% | 98.1% | 99.0% | 66.6% | 89.7% | 90.68% |
| Project recursive 600 | 88.7% | 94.8% | 97.2% | 69.6% | 84.7% | 89.24% |
| Page, postprocessed | 98.3% | 99.9% | 97.2% | 69.2% | 86.4% | 90.52% |
| LangChain recursive 1,100 | 97.7% | 98.6% | 98.4% | 64.7% | 86.8% | 88.37% |
| LangChain recursive default | 99.3% | 95.0% | 96.1% | 65.6% | 88.8% | 88.62% |
| Page, raw | 100.0% | 100.0% | 97.1% | 69.3% | 86.1% | 89.03% |
| Semantic, raw | 82.5% | 91.3% | 97.5% | 69.3% | 76.3% | 76.49% |
| Sentence, raw | 90.9% | 61.9% | 86.3% | 78.4% | 72.5% | 73.26% |
| Adaptive selection | **not run** | 99.4% | 99.0% | 68.2% | 88.8% | 91.07% |

Paper Table 4 selected postprocessed page for 48% of documents, recursive 1,100 for 42%, GPT-5 regex for 6%, and recursive 600 for 3%. **No Arabic selection percentages exist**, since three of the five required metrics are missing and BI is only a proxy.

## RAG and runtime (paper Tables 5 and 6)

| RAG measure | Arabic pilot | Paper adaptive | Paper LangChain default | Paper raw page |
|---|---:|---:|---:|---:|
| Retrieval completeness | Not measured | 67.68% | 58.08% | 59.09% |
| Answer correctness | Not measured | 78.01% | 70.11% | 73.33% |
| Paper combined mean | Not measured | 71.77% | 62.07% | 63.80% |
| Answered questions | No Arabic questions | 65 / 99 | 49 / 99 | 49 / 99 |

The pilot has no Arabic question/answer set, answer generation, or LLM judge. Paper Table 6 reports a **36:11** evaluation runtime; no equivalent Arabic five-metric evaluation was run, so no valid runtime comparison exists.

## Postprocessing effect (paper Table 7)

| Method | Arabic raw SC → final SC | Paper raw SC → final SC |
|---|---:|---:|
| Heading regex / GPT-5 regex | 68.6% → 99.9% | 58.3% → 99.6% |
| Project recursive 1,100 | 99.8%; no pilot postprocessing | 100.0% → 100.0% |
| Project recursive 600 | 99.5%; no pilot postprocessing | 98.7% → 100.0% |
| Page | 3.6% → 99.5% | 92.7% → 99.9% |
| LangChain recursive 1,100 | 93.4%; no pilot postprocessing | 93.3% → 99.4% |
| LangChain recursive default | 68.7%; no pilot postprocessing | 97.7%; no postprocessing |
| Semantic | 48.9% → 99.5% | 48.1% → 99.9% |
| Sentence | 79.5% → 99.9% | 67.2% → 100.0% |

The pilot establishes that the eight chunkers and size regularization can run on Arabic. It **does not establish that the paper's quality or RAG improvements transfer to Arabic**. A controlled follow-up needs natural Arabic reports, parser-derived structural blocks, validated Arabic embeddings and coreference links, and an Arabic question/answer set.
