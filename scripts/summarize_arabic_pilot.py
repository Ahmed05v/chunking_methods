"""Validate Arabic pilot chunk coverage and write a basic chunk-size report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adaptive_chunking.postprocessing import check_chunk_gaps


DATA_DIR = Path("data/arabic_pilot/adi_parsed")
RESULTS_DIR = Path("results/arabic_pilot")


def main() -> None:
    originals = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))["full_text"]
        for path in DATA_DIR.glob("*.json")
    }
    if len(originals) != 11:
        raise RuntimeError(f"Expected 11 Arabic documents, found {len(originals)}")

    rows = []
    for stage in ("raw", "no_oversizing", "small_merged"):
        path = RESULTS_DIR / "chunks" / stage / "chunks.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        for method, method_frame in frame.groupby("method"):
            covered = 0
            for name, group in method_frame.groupby("doc_name"):
                chunks = group.sort_values("chunk_index")["chunk_text"].tolist()
                covered += check_chunk_gaps(chunks, originals[name])
            lengths = method_frame["chunk_len"]
            rows.append({
                "stage": stage,
                "method": method,
                "documents": method_frame["doc_name"].nunique(),
                "documents_covered": covered,
                "chunks": len(method_frame),
                "mean_tokens": round(float(lengths.mean()), 1),
                "max_tokens": int(lengths.max()),
                "chunks_at_most_1100_pct": round(float((lengths <= 1100).mean() * 100), 1),
            })
    report = pd.DataFrame(rows).sort_values(["stage", "method"])
    output = RESULTS_DIR / "chunk_size_summary.csv"
    report.to_csv(output, index=False, encoding="utf-8")
    if not (report["documents"] == 11).all() or not (report["documents_covered"] == 11).all():
        raise RuntimeError("At least one method/stage failed document coverage")
    print(report.to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
