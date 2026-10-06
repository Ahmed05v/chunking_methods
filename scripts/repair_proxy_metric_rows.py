"""Remove legacy lexical scores accidentally saved as embedding metrics.

The older proxy pass wrote both `metric` and `metric_lexical_proxy` for the
same document/method. The latter identifies the former as a copied proxy.
"""
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results"
for subdir in ("results", "results_raw"):
    path = ROOT / subdir / "chunking_metrics.parquet"
    frame = pd.read_parquet(path)
    proxy = frame[frame.metric_name.str.endswith("_lexical_proxy")].copy()
    proxy["metric_name"] = proxy.metric_name.str.removesuffix("_lexical_proxy")
    keys = ["doc_name", "chunking_method", "metric_name"]
    copied = frame[keys].apply(tuple, axis=1).isin(set(proxy[keys].apply(tuple, axis=1)))
    copied &= ~frame.metric_name.str.endswith("_lexical_proxy")
    if copied.any():
        backup = path.with_name("chunking_metrics_before_proxy_repair.parquet")
        if not backup.exists():
            shutil.copy2(path, backup)
        frame.loc[~copied].to_parquet(path, index=False)
    print(f"{path}: removed {copied.sum()} mislabeled lexical scores")
