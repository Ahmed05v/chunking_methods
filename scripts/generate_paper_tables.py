"""Generate corpus comparison CSVs for paper Tables 1-7 from local artifacts."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def write_table4():
    from adaptive_chunking.paper.analysis import find_best_method
    metrics = pd.read_parquet(OUT / "results" / "chunking_metrics.parquet")
    chunks = pd.read_parquet(OUT / "chunks" / "small_merged" / "chunks.parquet")
    wanted = ["page", "our_recurs_1100", "our_recurs_600", "llm_regex"]
    weights = {"references_completeness": .2, "intrachunk_cohesion": .2,
               "document_contextual_coherence": .2, "block_integrity": .2,
               "size_compliance": .2}
    selected = {}
    for doc, frame in metrics[(metrics.chunking_method.isin(wanted)) & (~metrics.metric_name.str.endswith("_lexical_proxy"))].groupby("doc_name"):
        pivot = frame.pivot_table(index="metric_name", columns="chunking_method", values="score", aggfunc="mean")
        pivot = pivot.reindex(pivot.index.union(weights.keys()))
        available = [m for m in wanted if m in pivot and pivot[m].notna().any()]
        if not available:
            continue
        best, _ = find_best_method(pivot[available], weights)
        selected[doc] = best
    counts = pd.Series(selected).value_counts().reindex(wanted, fill_value=0)
    frame = pd.DataFrame({"method": wanted, "documents_selected": counts.values,
                          "share_percent": (counts.values / max(len(selected), 1) * 100).round(1)})
    frame["paper_share_percent"] = [48, 42, 3, 6]
    frame["note"] = "Local measured selection; not the paper's LLM output"
    frame.to_csv(OUT / "table4_method_selection.csv", index=False)
    pd.DataFrame([{"doc_name": k, "selected_method": v} for k, v in selected.items()]).to_csv(OUT / "table4_selection_per_document.csv", index=False)


def write_table5():
    qa_path = OUT / "rag" / "queries" / "generated_qa_pairs.json"
    if not qa_path.exists():
        return
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa_by_doc = {}
    for item in qa:
        name = item.get("doc_name", "")
        # Parsed source files are prefixed with a domain tag; generated QA names
        # omit it. Match the unique suffix while keeping all three QA items.
        qa_by_doc.setdefault(name, []).append(item)
    # RAG artifacts were not generated due missing API; create a retrieval proxy using
    # lexical TF-IDF search over each method's chunks and heuristic support scores.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    raw = pd.read_parquet(OUT / "chunks" / "raw" / "chunks.parquet")
    merged = pd.read_parquet(OUT / "chunks" / "small_merged" / "chunks.parquet")
    metrics = pd.read_parquet(OUT / "results" / "chunking_metrics.parquet")
    selected_methods = {}
    from adaptive_chunking.paper.analysis import find_best_method
    weights = {"references_completeness": .2, "intrachunk_cohesion": .2,
               "document_contextual_coherence": .2, "block_integrity": .2,
               "size_compliance": .2}
    for doc_name, frame in metrics[~metrics.metric_name.str.endswith("_lexical_proxy")].groupby("doc_name"):
        pivot = frame.pivot_table(index="metric_name", columns="chunking_method", values="score", aggfunc="mean")
        methods_here = [m for m in ["page", "our_recurs_1100", "our_recurs_600", "llm_regex"] if m in pivot and pivot[m].notna().any()]
        if methods_here:
            selected_methods[doc_name], _ = find_best_method(pivot[methods_here], weights)
    candidates = {"best": merged, "langch_recurs_default": raw, "page": raw}
    rows = []
    for method, df in candidates.items():
        if method == "best":
            chunks_by_doc = {}
            for k, selected in selected_methods.items():
                g = df[(df.doc_name == k) & (df.method == selected)].sort_values("chunk_index")
                if not g.empty:
                    chunks_by_doc[k] = g.chunk_text.astype(str).tolist()
        else:
            chunks_by_doc = {k: g.sort_values("chunk_index").chunk_text.astype(str).tolist() for k, g in df[df.method == method].groupby("doc_name")}
        hit, support, answer_overlap, count = 0, [], [], 0
        for qa_doc_name, pairs in qa_by_doc.items():
            matches = [name for name in chunks_by_doc if name.endswith("_" + qa_doc_name) or name == qa_doc_name]
            if not matches:
                continue
            corpus = chunks_by_doc[matches[0]]
            if not corpus:
                continue
            vec = TfidfVectorizer(stop_words="english", max_features=25000)
            try:
                matrix = vec.fit_transform(corpus)
            except ValueError:
                continue
            for pair in pairs:
                q = str(pair.get("question", ""))
                ans = str(pair.get("answer", pair.get("reference_answer", "")))
                qv = vec.transform([q])
                sims = cosine_similarity(qv, matrix).ravel()
                idx = int(np.argmax(sims))
                retrieved = corpus[idx]
                hit += int(sims[idx] > 0)
                answer_terms = {t.lower() for t in re.findall(r"\w+", ans) if len(t) > 2}
                retrieved_terms = {t.lower() for t in re.findall(r"\w+", retrieved) if len(t) > 2}
                overlap = len(answer_terms & retrieved_terms) / max(len(answer_terms), 1)
                answer_overlap.append(overlap)
                support.append(float(sims[idx] > 0) * overlap)
                count += 1
        retrieval_pct = 100 * hit / max(count, 1)
        correctness_pct = 100 * float(np.mean(support)) if support else None
        rows.append({"method": method, "questions": count, "retrieval_completeness_proxy_pct": round(retrieval_pct,2),
                     "correctness_lexical_proxy_pct": round(correctness_pct,2) if correctness_pct is not None else None,
                     "mean_proxy_pct": round((retrieval_pct + correctness_pct) / 2,2) if correctness_pct is not None else None,
                     "paper_mean_pct": {"best":71.77,"langch_recurs_default":62.07,"page":63.80}[method],
                     "note":"TF-IDF and reference-answer token overlap; not GPT-4.1 judged; replacement QA set"})
    pd.DataFrame(rows).to_csv(OUT / "table5_rag_proxy.csv", index=False)


def write_table6():
    chunk_perf = pd.read_parquet(OUT / "chunks" / "raw" / "performances.parquet")
    met_perf = pd.read_parquet(OUT / "results" / "metrics_performance.parquet")
    raw_perf = pd.read_parquet(OUT / "results_raw" / "metrics_performance.parquet")
    rows = []
    for method in ["page", "sentence", "langch_recurs_1100", "langch_recurs_default", "our_recurs_1100", "our_recurs_600", "semantic", "llm_regex"]:
        val = chunk_perf.loc[chunk_perf.method == method, "time"].sum()
        rows.append({"stage": "chunking:" + method, "seconds": float(val), "note": "Local wall times across processed docs"})
    # Add only non-embedding timings; embedding timing rows are incomplete due
    # interrupted high-memory runs and must not be mistaken for full runtimes.
    for name, df in [("postprocessed_metrics_non_embedding", met_perf), ("raw_metrics_non_embedding", raw_perf)]:
        non_embedding = df[~df.metric.isin(["chunk_embeddings", "intrachunk_cohesion", "document_contextual_coherence"])]
        rows.append({"stage": name, "seconds": float(non_embedding.time.sum()), "note": "Only recorded non-embedding portions; incomplete wall time"})
    pd.DataFrame(rows).to_csv(OUT / "table6_local_runtime.csv", index=False)


def write_table7():
    raw = pd.read_parquet(OUT / "chunks" / "raw" / "chunks.parquet")
    post = pd.read_parquet(OUT / "chunks" / "small_merged" / "chunks.parquet")
    methods = ["page", "sentence", "semantic", "llm_regex"]
    raw = raw[raw.chunk_len > 0]
    post = post[post.chunk_len > 0]
    a = raw[raw.method.isin(methods)].groupby("method").chunk_len.agg(raw_chunks="size", raw_mean="mean", raw_std="std", raw_min="min", raw_max="max")
    b = post[post.method.isin(methods)].groupby("method").chunk_len.agg(post_chunks="size", post_mean="mean", post_std="std", post_min="min", post_max="max")
    df = a.join(b, how="outer").reset_index()
    df["note"] = "Local raw vs postprocessed counts; local regex is heuristic approximation"
    df.to_csv(OUT / "table7_postprocessing_effect.csv", index=False)


def main():
    write_table4()
    write_table5()
    write_table6()
    write_table7()
    print("Wrote Tables 4-7 local comparison artifacts under results/.")


if __name__ == "__main__":
    main()
