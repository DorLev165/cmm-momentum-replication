"""
Momentum Trading - Characteristic-Managed Momentum (CMM)

Runs CMM on real S&P 500 data with expanding training window:
- Target: r_{i,t+1} (next-month returns), cross-sectionally normalized
- Loss: MSE between E_CMM signal and target
- Optimizer: Adam
- Window: train 1973-1982, test 1983-1984; expand 2 years, repeat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from cmm import prepare_cmm_data
from cmm.fetch_data import fetch_sp500_cmm_data
from cmm.portfolio import hml_summary, vol_managed_returns
from cmm.regime import regime_report
from cmm.training import run_expanding_window

# Data source: "jkp" (WRDS, full replication) or "yfinance" (quick, biased).
# Override with env var: CMM_DATA_SOURCE=jkp python main.py
DATA_SOURCE = os.environ.get("CMM_DATA_SOURCE", "yfinance").lower()


def plot_returns_and_variance(results: list, save_path: str = "cmm_returns_variance.png") -> None:
    """
    Plot HML cumulative return and rolling variance over time.
    """
    if not results or "hml_df" not in results[0]:
        return

    # Combine all monthly HML returns into one series (date index)
    frames = [r["hml_df"][["date", "HML_ret"]].set_index("date") for r in results]
    hml = pd.concat(frames).sort_index()
    hml = hml[~hml.index.duplicated(keep="first")]

    if len(hml) < 2:
        return

    # Cumulative return (growth of $1)
    cumret = (1 + hml["HML_ret"]).cumprod()

    # Rolling 12-month variance (annualized)
    roll_var = hml["HML_ret"].rolling(12, min_periods=1).var() * 12  # annualized

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(cumret.index, cumret.values, color="steelblue", linewidth=1.5)
    ax1.set_ylabel("Cumulative return ($1)")
    ax1.set_title("HML portfolio: cumulative return")
    ax1.axhline(1, color="gray", linestyle="--", alpha=0.7)
    ax1.grid(True, alpha=0.3)

    ax2.plot(roll_var.index, roll_var.values, color="coral", linewidth=1.2, alpha=0.9)
    ax2.set_ylabel("Variance (annualized)")
    ax2.set_xlabel("Year")
    ax2.set_title("HML portfolio: rolling 12-month variance")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")
    plt.show()


def main() -> None:
    """Run CMM with expanding training window on S&P 500 stocks."""
    print("=" * 60)
    print("CMM - Expanding Window (1973-1982 -> 1983-1984, +2y)")
    print("=" * 60)
    print("Loss: MSE(E_CMM, target) | Optimizer: Adam")
    print()

    # End-date cap (read here so the data fetch can use it; e.g.
    # CMM_END_DATE=2022-12-31 produces the paper-faithful sample).
    end_date_override = os.environ.get("CMM_END_DATE") or None

    # Fetch data. With DATA_SOURCE=jkp we pull the full JKP panel + CRSP
    # daily returns from WRDS; this is the paper's actual data and resolves
    # §1, §2, §3, §5, §6, §7 of CMM_REPLICATION_ISSUES.md.
    if DATA_SOURCE == "jkp":
        from cmm.fetch_data_jkp import fetch_jkp_cmm_data
        print("Data source: JKP / WRDS (full replication)")
        (
            characteristics,
            daily_returns,
            next_month_returns,
            dates,
            timestamps,
            tickers_arr,
            market_cap,
            is_nyse,
            regime_signal,
            industries,
        ) = fetch_jkp_cmm_data(
            start_date="1973-01-01",
            end_date=end_date_override,
            n_char=153,
            include_d_interactions=True,
        )
        n_char_used = characteristics.shape[1]
        train_start = 1973
        initial_train_end = 1982
        # JKP chars are already rank-transformed to [-0.5, 0.5]; don't re-scale.
        scale_features = False
    else:
        print("Data source: yfinance (S&P 500 only — survivorship-biased)")
        (
            characteristics,
            daily_returns,
            next_month_returns,
            dates,
            timestamps,
            tickers_arr,
            market_cap,
            is_nyse,
            regime_signal,
            industries,
        ) = fetch_sp500_cmm_data(
            start_date="1990-01-01",
            end_date=None,
            n_char=11,
            max_tickers=None,
        )
        n_char_used = 11
        train_start = 1990
        initial_train_end = 1999
        scale_features = True

    ts_min, ts_max = timestamps.min(), timestamps.max()
    print(f"  Data range: {pd.Timestamp(ts_min).year} - {pd.Timestamp(ts_max).year}")
    if DATA_SOURCE == "jkp":
        n_macro = n_char_used - 153 if n_char_used >= 153 else 0
        print(f"  Characteristics: {n_char_used} (153 JKP firm + {n_macro} macro)")
    else:
        print(f"  Characteristics: {n_char_used}")

    print("\nRunning expanding window (with portfolio construction)...")

    # Heavier model for full replication. Runtime scales roughly with
    # n_ensembles * epochs / batch_size. Rough budget on CPU:
    #   JKP universe: ~5-20 min per window per seed → 5 seeds × 20 windows = ~8-30h
    #   yfinance universe: ~30s per window per seed → ~5-15 min total
    # Drop n_ensembles to 1 and epochs to 30 for a fast smoke test.
    n_ensembles = int(os.environ.get("CMM_N_ENSEMBLES", "5"))
    epochs = int(os.environ.get("CMM_EPOCHS", "100"))
    # Toggles for the new machinery. 1 = on, 0 = off. Both default on for JKP.
    use_regime_ensemble = bool(int(os.environ.get("CMM_REGIME_ENSEMBLE", "1")))
    use_mv_portfolio = bool(int(os.environ.get("CMM_MV_PORTFOLIO", "1")))
    # Industry-adjusted returns (Lewellen 2015): subtract 2-digit-SIC industry
    # mean from each stock's next return before forming HML. Off by default.
    use_industry_adjust = bool(int(os.environ.get("CMM_INDUSTRY_ADJUST", "0")))
    # Output suffix lets parallel runs use separate directories so they
    # don't overwrite each other's plots and partial results.
    output_suffix = os.environ.get("CMM_OUTPUT_SUFFIX", "")
    plots_dir = f"plots{output_suffix}"
    results_dir = f"data/results{output_suffix}"
    print(
        f"  n_ensembles={n_ensembles}, epochs(max)={epochs}, "
        f"regime_ensemble={use_regime_ensemble}, mv_portfolio={use_mv_portfolio}, "
        f"industry_adjust={use_industry_adjust}"
    )
    if output_suffix:
        print(f"  output suffix: '{output_suffix}'  (plots → {plots_dir}, results → {results_dir})")
    if use_regime_ensemble:
        print("  (regime ensemble doubles training time — 2 sub-models per window)")

    results = run_expanding_window(
        characteristics,
        daily_returns,
        next_month_returns,
        dates,
        timestamps,
        tickers_arr,
        market_cap,
        is_nyse=is_nyse,
        regime_signal=regime_signal,
        industries=industries,
        use_regime_ensemble=use_regime_ensemble,
        use_mv_portfolio=use_mv_portfolio,
        use_industry_adjust=use_industry_adjust,
        n_char=n_char_used,
        n_ret=231,
        train_start_year=train_start,
        initial_train_end_year=initial_train_end,
        val_pct=0.3,            # paper uses last 30% of training months
        # Architecture — paper-faithful via Gu-Kelly-Xiu (2020) NN3.
        # The paper says "we follow Gu et al. (2020) for the architecture";
        # NN3 in that paper is (32, 16, 8), not the (256, 128, 64) we had
        # before. Our prior config had ~140k params; NN3 has ~13k. 10x
        # smaller, less prone to overfitting, much faster to train.
        hidden_sizes=(32, 16, 8),
        dropout=0.0,
        layer_norm=False,
        output_init_scale=1.5,
        # Optimization
        epochs=epochs,
        batch_size=4096,        # was 512 — closer to GKX convention (~10k)
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        lr_schedule="cosine",
        warmup_epochs=3,
        # Early stopping
        early_stopping_patience=10,
        min_epochs=15,
        # Loss
        loss_fn="mse",
        # Ensembling
        n_ensembles=n_ensembles,
        # Misc
        scale_features=scale_features,
        # Per-window persistence: write each window's result to disk so a
        # kill mid-run preserves work. Inspect with scripts/show_partial_results.py.
        partial_save_dir=results_dir,
        weights_plot_dir=plots_dir,
    )

    if not results:
        print("  No valid windows (data may not extend to 1973).")
        print("  Try a more recent start_date or check data availability.")
        return

    print("\n" + "-" * 80)
    print("Results by test period (VW = paper / MV = mean-variance within decile):")
    print("-" * 80)
    ics, hml_rets = [], []
    any_mv = any(r.get("hml_stats_mv") for r in results)
    for r in results:
        stats = r.get("hml_stats", {})
        stats_mv = r.get("hml_stats_mv", {}) or {}
        vw_sh = stats.get("sharpe", 0)
        vw_an = stats.get("ann_ret", 0) * 100
        vw_mdd = stats.get("mdd", 0) * 100
        line = (
            f"  {r['train_start']}-{r['train_end']} / {r['test_start']}-{r['test_end']} | "
            f"VW Sh {vw_sh:+.2f} AnnR {vw_an:+.1f}% MDD {vw_mdd:+.1f}%"
        )
        if any_mv and stats_mv:
            mv_sh = stats_mv.get("sharpe", 0)
            mv_an = stats_mv.get("ann_ret", 0) * 100
            mv_mdd = stats_mv.get("mdd", 0) * 100
            line += f" | MV Sh {mv_sh:+.2f} AnnR {mv_an:+.1f}% MDD {mv_mdd:+.1f}%"
        line += f" | IC {r['ic']:+.4f}"
        print(line)
        ics.append(r["ic"])
        if "mean_ret" in stats:
            hml_rets.append(stats["mean_ret"])

    print("-" * 80)
    # Pooled HML return series across all test periods (paper's Table 1 metric).
    all_hml = None
    all_hml_mv = None
    if hml_rets:
        all_hml = pd.concat(
            [r["hml_df"][["date", "HML_ret"]].set_index("date") for r in results if "hml_df" in r]
        ).sort_index()
        pooled = hml_summary(all_hml["HML_ret"])
        print(
            f"  POOLED HML (VW, paper-faithful): Sharpe {pooled['sharpe']:+.2f} | "
            f"AnnRet {pooled['ann_ret']*100:+.1f}% | "
            f"Vol {pooled['vol']*100:.1f}% | "
            f"MDD {pooled['mdd']*100:+.1f}% | "
            f"n={pooled['n_months']} months"
        )

    if any_mv:
        mv_dfs = [r.get("hml_df_mv") for r in results if r.get("hml_df_mv") is not None]
        if mv_dfs:
            all_hml_mv = pd.concat(
                [df[["date", "HML_ret"]].set_index("date") for df in mv_dfs]
            ).sort_index()
            pooled_mv = hml_summary(all_hml_mv["HML_ret"])
            print(
                f"  POOLED HML (MV, cov-aware):      Sharpe {pooled_mv['sharpe']:+.2f} | "
                f"AnnRet {pooled_mv['ann_ret']*100:+.1f}% | "
                f"Vol {pooled_mv['vol']*100:.1f}% | "
                f"MDD {pooled_mv['mdd']*100:+.1f}% | "
                f"n={pooled_mv['n_months']} months"
            )

    print(f"  Mean IC: {np.mean(ics):+.4f}  |  Std: {np.std(ics):.4f}")

    # Regime-conditional breakdown. The momentum_crash rows are the most
    # diagnostic for the paper's crash-resistance claim.
    if all_hml is not None and len(all_hml) > 0:
        # Volatility-managed HML — the lightweight, robust way to get
        # crash resistance without modifying the cross-sectional model.
        # (Daniel-Moskowitz 2016, Barroso & Santa-Clara 2015)
        scaled, leverage = vol_managed_returns(
            all_hml["HML_ret"], target_ann_vol=0.12, lookback_months=6, max_leverage=3.0,
        )
        scaled_summary = hml_summary(scaled)
        print()
        print(
            f"  VOL-MANAGED HML (target 12% ann vol, trailing 6mo, max 3x lev):"
        )
        print(
            f"    Sharpe {scaled_summary.get('sharpe', 0):+.2f} | "
            f"AnnRet {scaled_summary.get('ann_ret', 0)*100:+.1f}% | "
            f"Vol {scaled_summary.get('vol', 0)*100:.1f}% | "
            f"MDD {scaled_summary.get('mdd', 0)*100:+.1f}% | "
            f"Avg lev {leverage.mean():.2f}x"
        )

        print("\n" + "=" * 60)
        print("REGIME-CONDITIONAL HML PERFORMANCE")
        print("=" * 60)
        regime_series: list[tuple[str, pd.Series]] = [
            ("VW (value-weighted, paper)", all_hml["HML_ret"]),
            ("VW + vol-managed overlay", scaled),
        ]
        if all_hml_mv is not None and len(all_hml_mv) > 0:
            regime_series.append(("MV (cov-aware, LW shrinkage)", all_hml_mv["HML_ret"]))
        for series_name, series in regime_series:
            print(f"\n--- {series_name} ---")
            report = regime_report(series)
            for name, df in report.items():
                print(f"\n  [{name}]")
                if df is None or df.empty:
                    print("    (no data)")
                    continue
                print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPortfolio: deciles by CMM signal (NYSE breakpoints), value-weight (market cap),")
    print("HML explicitly dollar-neutral (long +1, short -1).")

    # Plot return and variance over time
    plot_returns_and_variance(results, save_path="cmm_returns_variance.png")

    print("Done.")


if __name__ == "__main__":
    main()
