# Replacement QA set

`generated_qa_pairs.json` contains 99 newly authored English question–answer pairs, three for each of the 33 documents in `data/clair/adi_parsed/`. The questions and reference answers were written by Codex from those documents on 2026-10-04 and use the JSON field names expected by `run_retrieval_for_generated_questions`.

This is **not** the unpublished GPT-4.1 question set used for Table 5 of the LREC 2026 paper. Do not compare scores obtained from this set with the paper's Table 5 as an exact reproduction. Run the repository's `rag` step with `--reuse-qa` to use this file; otherwise that step regenerates it with OpenAI.

To build indexes and retrieve passages without an OpenAI key, add `--retrieval-only` to the `rag` step. Answer generation and LLM judging still need a configured provider.

On Windows, the public Qwen and Snowflake model files can be downloaded to ordinary local folders to avoid Hugging Face cache symlink permissions. Pass those folders with `--embedding-model .hf-models/Qwen3-Embedding-4B --reranker-model .hf-models/snowflake-arctic-embed-l-v2.0`.
