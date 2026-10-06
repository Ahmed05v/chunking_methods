"""Join measured Arabic pilot intrinsic scores with the paper's Table 3 values."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from adaptive_chunking.metrics import compute_block_integrity


ROOT = Path(__file__).resolve().parents[1] / "results" / "arabic_pilot"
DATA = ROOT.parents[1] / "data" / "arabic_pilot" / "adi_parsed"
ROWS = [
    # label, stage, method, paper RC, ICC, DCC, BI, SC, overall
    ("Heading regex / GPT-5 regex", "small_merged", "llm_regex", 98.0, 70.9, 82.4, 98.1, 99.6, 89.80),
    ("Project recursive 1100", "raw", "our_recurs_1100", 99.0, 66.6, 89.7, 98.1, 100.0, 90.68),
    ("Project recursive 600", "raw", "our_recurs_600", 97.2, 69.6, 84.7, 94.8, 100.0, 89.24),
    ("Page, postprocessed", "small_merged", "page", 97.2, 69.2, 86.4, 99.9, 99.9, 90.52),
    ("LangChain recursive 1100", "raw", "langch_recurs_1100", 98.4, 64.7, 86.8, 98.6, 93.3, 88.37),
    ("LangChain recursive default", "raw", "langch_recurs_default", 96.1, 65.6, 88.8, 95.0, 97.7, 88.62),
    ("Page, raw", "raw", "page", 97.1, 69.3, 86.1, 100.0, 92.7, 89.03),
    ("Semantic, raw", "raw", "semantic", 97.5, 69.3, 76.3, 91.3, 48.1, 76.49),
    ("Sentence, raw", "raw", "sentence", 86.3, 78.4, 72.5, 61.9, 67.2, 73.26),
]


def main() -> None:
    sizes = pd.read_csv(ROOT / "chunk_size_summary.csv")
    embedding = pd.read_parquet(ROOT / "embedding_metrics_qwen.parquet")
    if len(embedding) != 121 or embedding.doc_name.nunique() != 11:
        raise RuntimeError("Expected 121 completed embedding method/document pairs across 11 documents")
    documents = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in DATA.glob("*.json")
    }
    chunks_by_stage = {
        stage: pd.read_parquet(ROOT / "chunks" / stage / "chunks.parquet")
        for stage in ("raw", "small_merged")
    }
    per_doc = []
    for stage, frame in chunks_by_stage.items():
        for (name, method), group in frame.groupby(["doc_name", "method"]):
            group = group.sort_values("chunk_index")
            doc = documents[name]
            lengths = group.chunk_len
            sc = float(((lengths >= 100) & (lengths <= 1100)).mean())
            bi = compute_block_integrity(
                group.chunk_text.tolist(), doc["split_points"], doc["full_text"], 5,
            )
            per_doc.append({"stage": stage, "doc_name": name, "method": method,
                            "sc": sc, "bi_paragraph_proxy": bi})
    joined = pd.DataFrame(per_doc).merge(
        embedding, on=["stage", "doc_name", "method"], how="left", validate="one_to_one",
    )
    joined["four_metric_proxy"] = joined[[
        "sc", "bi_paragraph_proxy", "icc_qwen", "dcc_qwen",
    ]].mean(axis=1, skipna=False)
    joined.to_csv(ROOT / "intrinsic_per_document.csv", index=False)
    rows = []
    for label, stage, method, rc, icc, dcc, bi, sc, overall in ROWS:
        size_row = sizes[(sizes.stage == stage) & (sizes.method == method)].iloc[0]
        group = joined[(joined.stage == stage) & (joined.method == method)]
        rows.append({
            "method": label, "arabic_documents": 11,
            "arabic_rc": None,
            "arabic_icc_qwen_mean_pct": round(group.icc_qwen.mean() * 100, 1),
            "arabic_icc_documents": int(group.icc_qwen.notna().sum()),
            "arabic_dcc_qwen_mean_pct": round(group.dcc_qwen.mean() * 100, 1),
            "arabic_dcc_documents": int(group.dcc_qwen.notna().sum()),
            "arabic_bi_paragraph_proxy_pct": size_row.paragraph_integrity_proxy_pct,
            "arabic_sc_pct": size_row.size_compliance_pct,
            "arabic_four_metric_proxy_mean_pct": round(group.four_metric_proxy.mean() * 100, 1),
            "arabic_four_metric_documents": int(group.four_metric_proxy.notna().sum()),
            "arabic_five_metric_mean": None,
            "paper_rc_pct": rc, "paper_icc_pct": icc,
            "paper_dcc_pct": dcc, "paper_bi_pct": bi,
            "paper_sc_pct": sc, "paper_five_metric_mean_pct": overall,
        })
    output = pd.DataFrame(rows)
    path = ROOT / "intrinsic_vs_paper.csv"
    output.to_csv(path, index=False)
    candidates = joined[
        ((joined.stage == "raw") & joined.method.isin({"our_recurs_1100", "our_recurs_600"}))
        | ((joined.stage == "small_merged") & joined.method.isin({"page", "llm_regex"}))
    ].dropna(subset=["four_metric_proxy"])
    winners = candidates.sort_values("four_metric_proxy", ascending=False).drop_duplicates("doc_name")
    if winners.doc_name.nunique() != 11:
        raise RuntimeError("The provisional selection must include all 11 documents")
    winners[["doc_name", "method", "four_metric_proxy"]].to_csv(
        ROOT / "provisional_four_metric_selection.csv", index=False,
    )
    print(output.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
