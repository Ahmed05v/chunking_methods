"""Measure Arabic pilot ICC/DCC with the local Qwen embedding model.

These scores use synthetic paragraph boundaries and Qwen, not the paper's
parser spans and Jina model. Results are saved after every method/document pair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from adaptive_chunking.metrics import (
    compute_chunk_embeddings,
    compute_contextual_coherence,
    compute_intrachunk_cohesion,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "arabic_pilot" / "adi_parsed"
CHUNKS = ROOT / "results" / "arabic_pilot" / "chunks"
OUT = ROOT / "results" / "arabic_pilot" / "embedding_metrics_qwen.parquet"
MODEL = ROOT / ".hf-models" / "Qwen3-Embedding-0.6B"
STAGES = {
    "raw": {"our_recurs_1100", "our_recurs_600", "langch_recurs_1100",
            "langch_recurs_default", "page", "semantic", "sentence"},
    "small_merged": {"page", "llm_regex", "semantic", "sentence"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many pending document/method pairs")
    args = parser.parse_args()

    groups = []
    for stage, methods in STAGES.items():
        frame = pd.read_parquet(CHUNKS / stage / "chunks.parquet")
        frame = frame[frame.method.isin(methods)]
        for (name, method), group in frame.groupby(["doc_name", "method"]):
            groups.append((stage, name, method, group.sort_values("chunk_index")))
    groups.sort(key=lambda item: sum(map(len, item[3].chunk_text)))

    columns = ["stage", "doc_name", "method", "icc_qwen", "dcc_qwen", "seconds"]
    results = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=columns)
    done = set(zip(results.stage, results.doc_name, results.method))
    pending = [item for item in groups if item[:3] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"Pending Arabic ICC/DCC pairs: {len(pending)}", flush=True)
    if not pending:
        return

    torch.set_num_threads(12)
    model = SentenceTransformer(
        str(MODEL), trust_remote_code=True, device=args.device,
        model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )
    for index, (stage, name, method, group) in enumerate(pending, 1):
        started = time.monotonic()
        print(f"[{index}/{len(pending)}] {stage}/{name}/{method}: {len(group)} chunks", flush=True)
        chunks = group.chunk_text.astype(str).tolist()
        doc = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
        try:
            vectors = compute_chunk_embeddings(chunks, model, batch_size=args.batch_size)
            icc = compute_intrachunk_cohesion(
                chunks, doc["full_text"], doc["split_points"], model,
                chunk_embeddings=vectors, batch_size=args.batch_size,
            )
            dcc = compute_contextual_coherence(
                chunks, doc["full_text"], model, window_context_tokens=3000,
                chunk_embeddings=vectors, batch_size=args.batch_size,
            )
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        row = {"stage": stage, "doc_name": name, "method": method,
               "icc_qwen": icc, "dcc_qwen": dcc,
               "seconds": round(time.monotonic() - started, 1)}
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_parquet(OUT, index=False)
        icc_label = "n/a" if icc is None else f"{icc:.3f}"
        dcc_label = "n/a" if dcc is None else f"{dcc:.3f}"
        print(f"  ICC={icc_label} DCC={dcc_label} {row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
