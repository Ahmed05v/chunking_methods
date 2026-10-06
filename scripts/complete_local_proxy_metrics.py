"""Complete missing local metric rows with transparent CPU-only lexical proxies.

Scores are added only where the repository's embedding metrics are missing.
The proxy rows are labeled `*_lexical_proxy` and must not be presented as exact
reproductions of the paper's embedding-based ICC/DCC values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from adaptive_chunking.chunking_utils import count_tokens
from adaptive_chunking.metrics import compute_block_integrity, compute_size_compliance
from adaptive_chunking.postprocessing import find_chunks_start_and_end


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DOCS = ROOT / "data" / "clair" / "adi_parsed"
MENTIONS = ROOT / "data" / "clair" / "mentions"
RAW_METHODS = {"page", "sentence", "semantic", "langch_recurs_1100", "langch_recurs_default", "llm_regex"}


def locate_mentions(doc_name: str):
    for path in MENTIONS.glob("*.parquet"):
        frame = pd.read_parquet(path)
        if not frame.empty and str(frame.doc_name.iloc[0]) == doc_name:
            return frame.entity_pron_mentions.iloc[0]
    return None


def lexical_similarity(chunks: list[str], full_text: str, split_points: list[int]):
    """Return TF-IDF analogues for cohesion and context coherence."""
    if len(chunks) < 2:
        return np.nan, np.nan
    vectorizer = TfidfVectorizer(stop_words="english", max_features=25000)
    try:
        chunk_vecs = vectorizer.fit_transform(chunks)
    except ValueError:
        return np.nan, np.nan
    lengths = np.asarray([max(count_tokens(chunk), 1) for chunk in chunks], dtype=float)
    bounds = find_chunks_start_and_end(chunks, full_text)
    sentence_parts = []
    for chunk, (start, end) in zip(chunks, bounds):
        local = [point - start for point in split_points if start <= point < end]
        cuts = sorted({0, *local, end})
        sentence_parts.append([chunk[a:b] for a, b in zip(cuts, cuts[1:]) if chunk[a:b].strip()])
    flat = [s for parts in sentence_parts for s in parts]
    if not flat:
        return np.nan, np.nan
    sentence_vecs = vectorizer.transform(flat)
    cohesion = []
    pos = 0
    for idx, parts in enumerate(sentence_parts):
        n = len(parts)
        if n > 1:
            sims = sentence_vecs[pos:pos+n] @ chunk_vecs[idx].T
            cohesion.extend(sims.toarray().ravel().tolist())
        pos += n
    icc = float(np.mean(cohesion)) if cohesion else np.nan
    # Approximate the paper's 3000-token document windows with consecutive chunks.
    windows, window_chunk_ids = [], []
    i = 0
    while i < len(chunks):
        ids, words = [], 0
        j = i
        while j < len(chunks) and (not ids or words < 3000):
            ids.append(j)
            words += lengths[j]
            j += 1
        if len(ids) > 1:
            windows.append(" ".join(chunks[k] for k in ids))
            window_chunk_ids.append(ids)
        i = j if j > i else i + 1
    dcc = np.nan
    if windows:
        wv = vectorizer.transform(windows)
        sims = (wv @ chunk_vecs.T).toarray()
        vals = [sims[w, c] for w, ids in enumerate(window_chunk_ids) for c in ids]
        if vals:
            dcc = float(np.mean(vals))
    return icc, dcc


def main():
    for target, chunk_file, methods in [
        (OUT / "results", OUT / "chunks" / "small_merged" / "chunks.parquet", None),
        (OUT / "results_raw", OUT / "chunks" / "raw" / "chunks.parquet", RAW_METHODS),
    ]:
        metrics_path = target / "chunking_metrics.parquet"
        df = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame(columns=["doc_name", "chunking_method", "metric_name", "score"])
        chunks = pd.read_parquet(chunk_file)
        if methods:
            chunks = chunks[chunks.method.isin(methods)]
        records = []
        for doc_name, doc_chunks in chunks.groupby("doc_name"):
            source_path = DOCS / f"{doc_name}.json"
            if not source_path.exists():
                continue
            doc = json.loads(source_path.read_text(encoding="utf-8"))
            full_text, split_points = doc["full_text"], doc["split_points"]
            pairs = locate_mentions(doc_name)
            existing = set(zip(df.loc[df.doc_name == doc_name, "chunking_method"], df.loc[df.doc_name == doc_name, "metric_name"]))
            for method, group in doc_chunks.groupby("method"):
                group = group.sort_values("chunk_index")
                text_chunks = group.chunk_text.astype(str).tolist()
                lengths = group.chunk_len.astype(int).tolist()
                values = {
                    "size_compliance": compute_size_compliance(text_chunks, 1100, 100),
                    "block_integrity": compute_block_integrity(text_chunks, split_points, full_text, 5),
                    "num_chunks": len(text_chunks),
                    "avg_chunk_tokens": float(np.mean(lengths)),
                    "stddev_chunk_tokens": float(np.std(lengths)),
                    "max_chunk_tokens": int(max(lengths)),
                    "min_chunk_tokens": int(min(lengths)),
                }
                for metric, value in values.items():
                    if (method, metric) not in existing:
                        records.append({"doc_name": doc_name, "chunking_method": method, "metric_name": metric, "score": value})
                for metric in ("intrachunk_cohesion", "document_contextual_coherence"):
                    if (method, metric) in existing:
                        continue
                    icc, dcc = lexical_similarity(text_chunks, full_text, split_points)
                    value = icc if metric == "intrachunk_cohesion" else dcc
                    records.append({"doc_name": doc_name, "chunking_method": method, "metric_name": metric + "_lexical_proxy", "score": value})
                if pairs is not None and (method, "references_completeness") not in existing:
                    from adaptive_chunking.metrics import compute_filtered_missing_ref_error
                    error = compute_filtered_missing_ref_error(full_text, text_chunks, pairs)
                    val = None if error is None else 1.0 - error
                    records.append({"doc_name": doc_name, "chunking_method": method, "metric_name": "references_completeness", "score": val})
        if records:
            combined = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
            # Existing exact rows take precedence; deduplicate any resumed rows.
            combined = combined.drop_duplicates(["doc_name", "chunking_method", "metric_name"], keep="last")
            combined.to_parquet(metrics_path, index=False)
        print(f"{metrics_path}: wrote {len(records)} rows; now {len(pd.read_parquet(metrics_path))} rows")


if __name__ == "__main__":
    main()
