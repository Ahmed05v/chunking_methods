"""Write exact-vs-proxy comparison CSVs with explicit coverage metadata."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PAPER = {
    "our_recurs_1100": ("Our recursive (1100), post-processed", 90.68),
    "our_recurs_600": ("Our recursive (600), post-processed", 89.24),
    "page": ("Page, post-processed", 90.52),
    "llm_regex": ("LLM regex (local heuristic approximation)", 89.80),
    "langch_recurs_1100": ("LangChain recursive (1100), raw", 88.37),
    "langch_recurs_default": ("LangChain recursive (default), raw", 88.62),
    "page_raw": ("Page, raw", 89.03),
    "semantic": ("Semantic (local Qwen approximation)", 76.49),
    "sentence": ("Sentence, raw", 73.26),
}
METRIC_MAP = {
    "RC": "references_completeness", "ICC": "intrachunk_cohesion",
    "DCC": "document_contextual_coherence", "BI": "block_integrity",
    "SC": "size_compliance",
}


def make_table3():
    post = pd.read_parquet(OUT / "results" / "chunking_metrics.parquet")
    raw = pd.read_parquet(OUT / "results_raw" / "chunking_metrics.parquet")
    rows, proxy_rows = [], []
    for method, (label, paper_mean) in PAPER.items():
        source = raw if method in {"langch_recurs_1100", "langch_recurs_default", "page_raw", "semantic", "sentence"} else post
        source_method = "page" if method == "page_raw" else method
        frame = source[(source.chunking_method == source_method) & (~source.metric_name.str.endswith("_lexical_proxy"))]
        record = {"method": label, "paper_mean": paper_mean}
        values = []
        for short, metric in METRIC_MAP.items():
            series = frame.loc[frame.metric_name == metric, "score"].dropna()
            record[f"{short}_mean_pct"] = round(series.mean()*100, 2) if len(series) else np.nan
            record[f"{short}_std_pct"] = round(series.std()*100, 2) if len(series) > 1 else np.nan
            record[f"{short}_n"] = int(series.size)
            # Separate lexical proxy values (where present).
            proxy_frame = source.loc[(source.chunking_method == source_method) & (source.metric_name == metric + "_lexical_proxy")]
            exact_docs = set(frame.loc[(frame.metric_name == metric) & frame.score.notna(), "doc_name"])
            proxy = proxy_frame.loc[~proxy_frame.doc_name.isin(exact_docs), "score"].dropna()
            # Never blend lexical proxies into a paper-metric aggregate. They
            # are reported in the companion proxy CSV only.
            if len(proxy):
                proxy_rows.append({"method": label, "metric": short, "lexical_proxy_mean_pct": round(proxy.mean()*100, 2), "n": int(proxy.size), "note": "TF-IDF heuristic, not paper embedding metric"})
            values.append(float(series.mean()) if len(series) else np.nan)
        enough_coverage = all(record[f"{short}_n"] >= 30 for short in METRIC_MAP)
        record["local_mean_of_available_paper_metrics_pct"] = round(np.nanmean(values)*100, 2) if enough_coverage else np.nan
        record["local_minus_paper_mean_pp"] = round(record["local_mean_of_available_paper_metrics_pct"] - paper_mean, 2) if enough_coverage else np.nan
        record["note"] = "Partial embedding coverage; mean withheld" if not enough_coverage else "Local approximation for LLM regex/semantic; see per-metric n and proxy CSV"
        rows.append(record)
    pd.DataFrame(rows).to_csv(OUT / "table3_comparison.csv", index=False)
    pd.DataFrame(proxy_rows).to_csv(OUT / "table3_lexical_proxy_scores.csv", index=False)


def make_table2():
    chunks_post = pd.read_parquet(OUT / "chunks" / "small_merged" / "chunks.parquet")
    chunks_raw = pd.read_parquet(OUT / "chunks" / "raw" / "chunks.parquet")
    paper_path = OUT / "table2_comparison.csv"
    old = pd.read_csv(paper_path)
    rename = {"page": "Page, post-processed", "our_recurs_1100": "Our recursive (1100), post-processed", "our_recurs_600": "Our recursive (600), post-processed", "llm_regex": "LLM regex (local heading-based approximation)", "langch_recurs_1100": "LangChain recursive (1100), raw", "langch_recurs_default": "LangChain recursive (default), raw", "semantic": "Semantic (local Qwen model)", "sentence": "Sentence, raw"}
    rows=[]
    for key,label in rename.items():
        chunks = chunks_post if key in {"page", "our_recurs_1100", "our_recurs_600", "llm_regex"} else chunks_raw
        g=chunks[chunks.method==key]
        if not len(g): continue
        oldrow=old[old.method==label]
        paper_chunks=int(oldrow.paper_chunks.iloc[0]) if len(oldrow) else np.nan
        rows.append({"method":label,"our_chunks":len(g),"paper_chunks":paper_chunks,"our_mean_tokens":round(g.chunk_len.mean(),1),"our_max_tokens":int(g.chunk_len.max()),"our_min_tokens":int(g.chunk_len.min()),"our_std_tokens":round(g.chunk_len.std(ddof=0),1),"coverage_docs":int(g.doc_name.nunique()),"note":"LLM regex is deterministic local heuristic; semantic uses local Qwen" if key in {"llm_regex","semantic"} else ""})
    pd.DataFrame(rows).to_csv(OUT / "table2_comparison.csv",index=False)


if __name__ == "__main__":
    make_table3()
    make_table2()
