"""
Print a summary of partial results saved during a run.

After each window completes, `run_expanding_window` writes a JSON line to
data/results/results_partial.jsonl plus per-window HML returns CSVs.
This script reads those files and prints a per-window table + a pooled
summary across whatever windows are complete so far.

Use it to:
  - Check progress mid-run without killing the process.
  - Inspect partial results after a kill.

Usage:
    python scripts/show_partial_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_dir() -> Path:
    """
    Resolve the results directory. Defaults to data/results, but accepts a
    suffix as the first command-line argument so parallel runs can be
    inspected separately:

        python scripts/show_partial_results.py             -> data/results
        python scripts/show_partial_results.py _baseline   -> data/results_baseline
    """
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    base = Path(__file__).resolve().parent.parent / "data" / f"results{suffix}"
    return base


_RESULTS_DIR = _resolve_dir()
_JSONL = _RESULTS_DIR / "results_partial.jsonl"
_VW_DIR = _RESULTS_DIR / "hml_returns_vw"
_MV_DIR = _RESULTS_DIR / "hml_returns_mv"


def _hml_summary(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {}
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    return {
        "ann_ret": float(r.mean() * 12),
        "vol": float(r.std() * np.sqrt(12)),
        "sharpe": float(r.mean() / (r.std() + 1e-12) * np.sqrt(12)),
        "mdd": float(dd.min()),
        "n_months": int(len(r)),
    }


def main() -> None:
    if not _JSONL.exists():
        print(f"No partial results at {_JSONL}")
        print("Either no run has saved yet, or partial_save_dir wasn't set.")
        return

    records = []
    with open(_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [warn] skipping malformed line: {e}")

    if not records:
        print(f"{_JSONL} is empty.")
        return

    print(f"Loaded {len(records)} window results from {_JSONL}")
    has_mv = any(r.get("hml_stats_mv") for r in records)

    # ---- Per-window table ----
    print()
    print("-" * (105 if has_mv else 75))
    if has_mv:
        print(
            f"{'Window':<26} | "
            f"{'VW Sh':>7} {'AnnR%':>7} {'MDD%':>7} | "
            f"{'MV Sh':>7} {'AnnR%':>7} {'MDD%':>7} | "
            f"{'IC':>8}"
        )
    else:
        print(
            f"{'Window':<26} | "
            f"{'VW Sh':>7} {'AnnR%':>7} {'MDD%':>7} | "
            f"{'IC':>8}"
        )
    print("-" * (105 if has_mv else 75))

    for r in records:
        vw = r.get("hml_stats", {}) or {}
        mv = r.get("hml_stats_mv", {}) or {}
        win = f"{r['train_start']}-{r['train_end']}/{r['test_start']}-{r['test_end']}"
        if has_mv:
            print(
                f"{win:<26} | "
                f"{vw.get('sharpe', 0):+7.2f} "
                f"{vw.get('ann_ret', 0)*100:+7.1f} "
                f"{vw.get('mdd', 0)*100:+7.1f} | "
                f"{mv.get('sharpe', 0):+7.2f} "
                f"{mv.get('ann_ret', 0)*100:+7.1f} "
                f"{mv.get('mdd', 0)*100:+7.1f} | "
                f"{r.get('ic', 0):+8.4f}"
            )
        else:
            print(
                f"{win:<26} | "
                f"{vw.get('sharpe', 0):+7.2f} "
                f"{vw.get('ann_ret', 0)*100:+7.1f} "
                f"{vw.get('mdd', 0)*100:+7.1f} | "
                f"{r.get('ic', 0):+8.4f}"
            )

    print("-" * (105 if has_mv else 75))

    # ---- Pooled stats from CSVs (re-aggregated from monthly returns) ----
    def _pool(directory: Path) -> pd.Series:
        if not directory.exists():
            return pd.Series(dtype=float)
        dfs = []
        for csv in sorted(directory.glob("*.csv")):
            try:
                df = pd.read_csv(csv, parse_dates=["date"])
                dfs.append(df[["date", "HML_ret"]].set_index("date"))
            except Exception:
                continue
        if not dfs:
            return pd.Series(dtype=float)
        return pd.concat(dfs).sort_index()["HML_ret"]

    pooled_vw = _pool(_VW_DIR)
    if len(pooled_vw) > 0:
        s = _hml_summary(pooled_vw)
        print(
            f"\n  POOLED VW: Sharpe {s['sharpe']:+.2f} | "
            f"AnnRet {s['ann_ret']*100:+.1f}% | "
            f"Vol {s['vol']*100:.1f}% | "
            f"MDD {s['mdd']*100:+.1f}% | "
            f"n={s['n_months']} months"
        )

    if has_mv:
        pooled_mv = _pool(_MV_DIR)
        if len(pooled_mv) > 0:
            s = _hml_summary(pooled_mv)
            print(
                f"  POOLED MV: Sharpe {s['sharpe']:+.2f} | "
                f"AnnRet {s['ann_ret']*100:+.1f}% | "
                f"Vol {s['vol']*100:.1f}% | "
                f"MDD {s['mdd']*100:+.1f}% | "
                f"n={s['n_months']} months"
            )

    ics = [r.get("ic", 0) for r in records]
    print(f"\n  Mean IC across windows: {np.mean(ics):+.4f}  |  Std: {np.std(ics):.4f}")


if __name__ == "__main__":
    main()
