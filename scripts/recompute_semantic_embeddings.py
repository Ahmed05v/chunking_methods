"""Resume exact Jina ICC/DCC scores for raw semantic chunks, one document at a time."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from adaptive_chunking.metrics import (
    compute_chunk_embeddings,
    compute_contextual_coherence,
    compute_intrachunk_cohesion,
)
from adaptive_chunking.paper.replicate import _make_embedder

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCORES = RESULTS / "results_raw" / "chunking_metrics.parquet"
CHUNKS = RESULTS / "chunks" / "raw" / "chunks.parquet"
DOCS = ROOT / "data" / "clair" / "adi_parsed"
ICC = "intrachunk_cohesion"
DCC = "document_contextual_coherence"


def existing_score(frame: pd.DataFrame, doc: str, metric: str) -> bool:
    rows = frame[(frame.doc_name == doc) & (frame.chunking_method == "semantic") & (frame.metric_name == metric)]
    return bool(rows.score.notna().any())


def save_score(frame: pd.DataFrame, doc: str, metric: str, value: float | None) -> pd.DataFrame:
    key = (frame.doc_name == doc) & (frame.chunking_method == "semantic") & (frame.metric_name == metric)
    frame = frame.loc[~key].copy()
    row = pd.DataFrame([{"doc_name": doc, "chunking_method": "semantic", "metric_name": metric, "score": value}])
    frame = pd.concat([frame, row], ignore_index=True)
    frame.to_parquet(SCORES, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only process this many pending documents")
    args = parser.parse_args()

    torch.set_num_threads(12)
    frame = pd.read_parquet(SCORES)
    chunks = pd.read_parquet(CHUNKS)
    chunks = chunks[chunks.method == "semantic"]
    groups = sorted(chunks.groupby("doc_name"), key=lambda item: sum(map(len, item[1].chunk_text)))
    pending = [(doc, group) for doc, group in groups if not all(existing_score(frame, doc, metric) for metric in (ICC, DCC))]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"Exact semantic ICC/DCC pending for {len(pending)} documents", flush=True)
    if not pending:
        return
    model = _make_embedder(args.device)
    errors = RESULTS / "semantic_exact_errors.txt"
    for number, (doc_name, group) in enumerate(pending, 1):
        started = time.monotonic()
        print(f"[{number}/{len(pending)}] {doc_name} ({len(group)} chunks)", flush=True)
        try:
            group = group.sort_values("chunk_index")
            texts = group.chunk_text.astype(str).tolist()
            parsed = json.loads((DOCS / f"{doc_name}.json").read_text(encoding="utf-8"))
            chunk_vectors = compute_chunk_embeddings(texts, model, batch_size=args.batch_size)
            if not existing_score(frame, doc_name, ICC):
                score = compute_intrachunk_cohesion(
                    chunks=texts, full_text=parsed["full_text"], split_points=parsed["split_points"],
                    model=model, chunk_embeddings=chunk_vectors, batch_size=args.batch_size,
                )
                frame = save_score(frame, doc_name, ICC, score)
                print(f"  ICC={score} saved after {time.monotonic()-started:.1f}s", flush=True)
            if not existing_score(frame, doc_name, DCC):
                score = compute_contextual_coherence(
                    chunks=texts, full_text=parsed["full_text"], model=model,
                    chunk_embeddings=chunk_vectors, window_context_tokens=3000,
                    window_step=1, batch_size=args.batch_size,
                )
                frame = save_score(frame, doc_name, DCC, score)
                print(f"  DCC={score} saved after {time.monotonic()-started:.1f}s", flush=True)
        except Exception as exc:
            with errors.open("a", encoding="utf-8") as stream:
                stream.write(f"{doc_name}: {type(exc).__name__}: {exc}\n")
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
