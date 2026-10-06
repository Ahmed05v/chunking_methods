"""Build a readable paper-versus-local report from the current result files."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PAPER = "https://aclanthology.org/2026.lrec-1.903.pdf"

# Table 3, in paper order: RC, ICC, DCC, BI, SC (mean, standard deviation).
PAPER_METRICS = {
    "LLM regex (local heuristic approximation)": [(98.0,2.9),(70.9,5.1),(82.4,5.5),(98.1,2.5),(99.6,1.3)],
    "Our recursive (1100), post-processed": [(99.0,1.8),(66.6,3.8),(89.7,2.5),(98.1,1.9),(100.0,0.1)],
    "Our recursive (600), post-processed": [(97.2,2.9),(69.6,4.1),(84.7,3.1),(94.8,3.6),(100.0,0.0)],
    "Page, post-processed": [(97.2,3.5),(69.2,3.6),(86.4,4.6),(99.9,0.3),(99.9,0.4)],
    "LangChain recursive (1100), raw": [(98.4,3.1),(64.7,3.4),(86.8,2.8),(98.6,1.4),(93.3,7.2)],
    "LangChain recursive (default), raw": [(96.1,4.3),(65.6,3.6),(88.8,2.4),(95.0,2.8),(97.7,6.3)],
    "Page, raw": [(97.1,3.5),(69.3,3.4),(86.1,5.0),(100.0,0.0),(92.7,9.7)],
    "Semantic (local Qwen approximation)": [(97.5,3.1),(69.3,4.1),(76.3,7.3),(91.3,4.0),(48.1,17.6)],
    "Sentence, raw": [(86.3,11.1),(78.4,2.7),(72.5,5.8),(61.9,10.3),(67.2,23.1)],
    "Adaptive selection (local)": [(99.0,1.5),(68.2,4.7),(88.8,3.5),(99.4,1.2),(99.9,0.3)],
}
PAPER_CHUNKS = {
    "LLM regex (local heuristic approximation)": (2279,518),
    "Our recursive (1100), post-processed": (1345,878),
    "Our recursive (600), post-processed": (2381,496),
    "Page, post-processed": (1780,663),
    "LangChain recursive (1100), raw": (1675,706),
    "LangChain recursive (default), raw": (1557,773),
    "Page, raw": (1765,669),
    "Semantic (local Qwen approximation)": (1706,693),
    "Sentence, raw": (8125,146),
    "Adaptive selection (local)": (1631,724),
}
CHUNK_KEYS = {
    "LLM regex (local heuristic approximation)": ("small_merged","llm_regex"),
    "Our recursive (1100), post-processed": ("small_merged","our_recurs_1100"),
    "Our recursive (600), post-processed": ("small_merged","our_recurs_600"),
    "Page, post-processed": ("small_merged","page"),
    "LangChain recursive (1100), raw": ("raw","langch_recurs_1100"),
    "LangChain recursive (default), raw": ("raw","langch_recurs_default"),
    "Page, raw": ("raw","page"),
    "Semantic (local Qwen approximation)": ("raw","semantic"),
    "Sentence, raw": ("raw","sentence"),
}
METRICS = ["RC","ICC","DCC","BI","SC"]


def adaptive_row() -> dict:
    selected = pd.read_csv(OUT / "table4_selection_per_document.csv")
    metrics = pd.read_parquet(OUT / "results" / "chunking_metrics.parquet")
    merged = metrics.merge(selected, on="doc_name")
    merged = merged[merged.chunking_method == merged.selected_method]
    mapping = {"RC":"references_completeness","ICC":"intrachunk_cohesion",
               "DCC":"document_contextual_coherence","BI":"block_integrity","SC":"size_compliance"}
    row = {"method":"Adaptive selection (local)","paper_mean":91.07}
    for short, metric in mapping.items():
        series = merged.loc[merged.metric_name == metric, "score"].dropna()
        row[f"{short}_mean_pct"] = series.mean()*100
        row[f"{short}_std_pct"] = series.std()*100
        row[f"{short}_n"] = len(series)
    row["local_mean_of_available_paper_metrics_pct"] = sum(row[f"{k}_mean_pct"] for k in METRICS)/5
    return row


def render() -> str:
    df = pd.read_csv(OUT / "table3_comparison.csv")
    rows = {row.method:row.to_dict() for _,row in df.iterrows()}
    rows["Adaptive selection (local)"] = adaptive_row()
    lines = [
        "# Final local results versus the paper",
        "",
        f"Source: [Adaptive Chunking, LREC 2026]({PAPER}), Tables 1–7. Local files: `results/`.",
        "",
        "**RC** = reference completeness; **ICC** = cohesion within chunks; **DCC** = agreement with nearby document context; **BI** = intact structural blocks; **SC** = chunks within the 100–1,100 token size range. All scores are percentages. A cell is **local mean ± standard deviation / paper mean ± standard deviation**.",
        "",
        "| Method | RC (local / paper) | ICC (local / paper) | DCC (local / paper) | BI (local / paper) | SC (local / paper) | Overall local / paper | Δ points |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, paper in PAPER_METRICS.items():
        row=rows[name]
        cells=[]
        for key,(mean,std) in zip(METRICS,paper):
            cells.append(f"{row[f'{key}_mean_pct']:.1f}±{row[f'{key}_std_pct']:.1f} / {mean:.1f}±{std:.1f}")
        local=row["local_mean_of_available_paper_metrics_pct"]
        paper_mean=row["paper_mean"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | **{local:.2f} / {paper_mean:.2f}** | {local-paper_mean:+.2f} |")
    lines += [
        "",
        "The LLM regex row is a local heading rule in our run; the paper used GPT-5. The semantic row uses our local Qwen chunker, then Jina v3 for **all 33 real ICC and DCC scores**. RC has 31 available English documents; other individual scores have between 30 and 33 valid documents. See `table3_comparison.csv` for each score's exact sample count. Adaptive selection uses our locally selected method per document, so its row can differ from the paper's selection.",
        "",
        "## Chunk sizes (paper Table 2)",
        "",
        "| Method | Local chunks / paper chunks | Mean tokens: local / paper |",
        "|---|---:|---:|",
    ]
    chunks={stage:pd.read_parquet(OUT / "chunks" / stage / "chunks.parquet") for stage in ("raw","small_merged")}
    selected=pd.read_csv(OUT / "table4_selection_per_document.csv")
    adaptive=chunks["small_merged"].merge(selected,on="doc_name")
    adaptive=adaptive[adaptive.method==adaptive.selected_method]
    for name in PAPER_METRICS:
        if name=="Adaptive selection (local)":
            sub=adaptive
        else:
            stage,method=CHUNK_KEYS[name]
            sub=chunks[stage][chunks[stage].method==method]
        paper_n,paper_avg=PAPER_CHUNKS[name]
        lines.append(f"| {name} | {len(sub):,} / {paper_n:,} | {sub.chunk_len.mean():.0f} / {paper_avg} |")
    lines += [
        "",
        "## Corpus (paper Table 1)",
        "",
        "| Domain | Documents: local / paper | Total tokens: local / paper | Mean pages: local / paper |",
        "|---|---:|---:|---:|",
    ]
    corpus=pd.read_csv(OUT / "table1_corpus_comparison.csv")
    for _,r in corpus.iterrows():
        lines.append(f"| {r.domain} | {int(r.our_docs)} / {int(r.paper_docs)} | {int(r.our_total_tokens):,} / {int(r.paper_total_tokens):,} | {int(r.our_mean_pages)} / {int(r.paper_mean_pages)} |")
    lines += [
        "",
        "## Size compliance before and after processing (paper Table 7)",
        "",
        "| Method | Raw SC: local / paper | Final SC: local / paper |",
        "|---|---:|---:|",
    ]
    sc_paper={
        "LLM regex (local heuristic approximation)":(58.3,99.6),
        "Our recursive (1100), post-processed":(100.0,100.0),
        "Our recursive (600), post-processed":(98.7,100.0),
        "Page, post-processed":(92.7,99.9),
        "LangChain recursive (1100), raw":(93.3,99.4),
        "LangChain recursive (default), raw":(97.7,None),
        "Semantic (local Qwen approximation)":(48.1,99.9),
        "Sentence, raw":(67.2,100.0),
    }
    from adaptive_chunking.metrics import compute_size_compliance
    metric_frames={stage:pd.read_parquet(OUT / folder / "chunking_metrics.parquet") for stage,folder in (("raw","results_raw"),("small_merged","results"))}
    for name,(raw_paper,post_paper) in sc_paper.items():
        method=CHUNK_KEYS[name][1]
        raw_chunks=chunks["raw"][chunks["raw"].method==method]
        post_chunks=chunks["small_merged"][chunks["small_merged"].method==method]
        def sc(stage, subset):
            scores=metric_frames[stage]
            values=scores[(scores.chunking_method==method)&(scores.metric_name=="size_compliance")].score.dropna()
            if not values.empty:
                return values.mean()*100
            return pd.Series({doc:compute_size_compliance(g.sort_values("chunk_index").chunk_text.astype(str).tolist()) for doc,g in subset.groupby("doc_name")}).mean()*100
        raw_sc=sc("raw",raw_chunks)
        post_sc=sc("small_merged",post_chunks)
        post_cell=f"{post_sc:.1f}% / {post_paper:.1f}%" if post_paper is not None else "n/a"
        lines.append(f"| {name} | {raw_sc:.1f}% / {raw_paper:.1f}% | {post_cell} |")
    lines += ["", "These local SC values average each document's size-compliance score. The paper's full Table 7 also reports mean quality scores at the intermediate oversized-split stage, which we have not fully rescored."]
    lines += [
        "",
        "## Adaptive method choice (paper Table 4)",
        "",
        "| Selected method | Local documents | Local share | Paper share |",
        "|---|---:|---:|---:|",
    ]
    selections=pd.read_csv(OUT / "table4_method_selection.csv")
    for _,r in selections.iterrows():
        lines.append(f"| {r.method} | {int(r.documents_selected)} | {r.share_percent:.1f}% | {r.paper_share_percent:.0f}% |")
    lines += [
        "",
        "## RAG evaluation (paper Table 5)",
        "",
        "These are **different tests**. The paper used its GPT-4.1 generated questions, answer generation, and GPT-4.1 judging. Our 99 questions were created separately; TF-IDF retrieved chunks and word overlap gave a simple proxy. The proxy is **not** an estimate of GPT-4.1 answer correctness or retrieval completeness.",
        "",
        "| Method | Our lexical retrieval hit* | Our answer-word overlap* | Paper retrieval completeness | Paper answer correctness | Paper mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rag=pd.read_csv(OUT / "table5_rag_proxy.csv")
    paper_rag={"best":(67.68,78.01,71.77),"langch_recurs_default":(58.08,70.11,62.07),"page":(59.09,73.33,63.80)}
    for _,r in rag.iterrows():
        pr,pc,pm=paper_rag[r.method]
        lines.append(f"| {r.method} | {r.retrieval_completeness_proxy_pct:.1f}% | {r.correctness_lexical_proxy_pct:.1f}% | {pr:.2f}% | {pc:.2f}% | {pm:.2f}% |")
    lines += [
        "",
        "*Local retrieval hit means the best TF-IDF match was above zero; answer-word overlap measures how many reference-answer words appear in that chunk. They are diagnostic proxies, not the paper's metrics. The paper reports 65 answered queries for adaptive selection and 49 for each baseline; our proxy did not generate answers.",
        "",
        "## Models and tools used",
        "",
        "| Task | Paper | Our run |",
        "|---|---|---|",
        "| Documents and parsing | 33 CLAIR documents with parsed text and structural boundaries | Same 33 parsed source documents |",
        "| LLM regex chunks | GPT-5 selects a regex from a document sample | Fixed local heading-based regex; no LLM call |",
        "| Semantic chunks | LangChain experimental semantic splitter with gradient thresholding | Same splitter type with local Qwen3-Embedding-0.6B; raw chunks used for Table 3 |",
        "| Recursive and page chunks | Authors' recursive splitter, LangChain recursive splitter, page splitter | Repository implementations of those splitters |",
        "| Sentence chunks | Stanza, five sentences per chunk | Stanza, five sentences per chunk |",
        "| ICC and DCC scoring | Jina `jina-embeddings-v3` sentence embeddings | Local Jina `jina-embeddings-v3`; semantic ICC and DCC rerun across 33 documents |",
        "| Reference links (RC) | Maverick `maverick-mes-ontonotes` entity/pronoun extraction | Precomputed Maverick mention files; 31 documents yield RC scores |",
        "| Tokens, size, structural integrity | OpenAI `o200k_base` token encoding; parser block boundaries | `tiktoken` and the parsed block boundaries |",
        "| RAG questions | GPT-4.1, three per document | 99 separately authored replacement questions, three per document |",
        "| RAG retrieval | Qwen3-Embedding-4B dense search + BM25 + Snowflake reranking | Local TF-IDF retrieval proxy |",
        "| RAG answers and scoring | GPT-4.1 generation and GPT-4.1/DeepEval judging | Answer-word overlap proxy; no generative model or LLM judge |",
        "",
        "The paper's exact RAG Table 5 remains unreproduced because its original question set and GPT-4.1 evaluation were not used here. Table 6 local metric runtimes are partial after resumed runs and cannot be compared fairly with the paper's 36:11 total. `table7_postprocessing_effect.csv` contains the local raw-versus-postprocessed chunk-size data; the paper's Table 7 also reports scores at intermediate stages, which were not all recomputed locally.",
        "",
    ]
    return "\n".join(lines)


if __name__=="__main__":
    target=OUT / "FINAL_COMPARISON.md"
    target.write_text(render(),encoding="utf-8")
    print(target)
