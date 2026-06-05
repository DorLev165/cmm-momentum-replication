@echo off
REM ============================================================
REM Combined CMM + EW-momentum pipeline
REM
REM This runs in two stages:
REM
REM   Stage 1  (~3-5h): Run the CMM strategy with the proven-best
REM                     config — full universe, no regime ensemble,
REM                     no industry adjust, no MV portfolio, 5 seeds.
REM                     This regenerates the bear-Sharpe-positive CMM
REM                     data we previously had but lost (the persistence
REM                     code didn't exist when that run completed).
REM
REM   Stage 2  (~30s):  Combine the fresh CMM monthly returns with
REM                     friend's EW-momentum NAVs using the bias-free
REM                     regime classifier (point-in-time signals,
REM                     fixed thresholds). Produces final results.
REM
REM Run from project root:
REM     scripts\run_combined.bat
REM ============================================================

setlocal

REM ----- Configuration shared across both stages -----
set CMM_OUTPUT_SUFFIX=_combined
set CMM_DATA_SUFFIX=_combined

REM ----- Stage 1: CMM run with paper-faithful config -----
echo.
echo ============================================================
echo Stage 1: CMM training pipeline (3-5 hours expected)
echo ============================================================

set CMM_DATA_SOURCE=jkp
set CMM_REGIME_ENSEMBLE=0
set CMM_MV_PORTFOLIO=0
set CMM_N_ENSEMBLES=5
set CMM_INDUSTRY_ADJUST=0
set CMM_END_DATE=2022-12-31

echo Config:
echo   CMM_DATA_SOURCE=%CMM_DATA_SOURCE%
echo   CMM_REGIME_ENSEMBLE=%CMM_REGIME_ENSEMBLE%
echo   CMM_MV_PORTFOLIO=%CMM_MV_PORTFOLIO%
echo   CMM_N_ENSEMBLES=%CMM_N_ENSEMBLES%
echo   CMM_INDUSTRY_ADJUST=%CMM_INDUSTRY_ADJUST%
echo   CMM_END_DATE=%CMM_END_DATE%
echo   CMM_OUTPUT_SUFFIX=%CMM_OUTPUT_SUFFIX%
echo.

python "C:\Projects\Momentum trading\main.py"

if errorlevel 1 (
    echo.
    echo [BLOCKER] CMM pipeline failed. Stage 2 will not run.
    exit /b 1
)

echo.
echo Stage 1 complete. CMM results written to:
echo   data\results%CMM_OUTPUT_SUFFIX%\
echo   plots%CMM_OUTPUT_SUFFIX%\

REM ----- Stage 2: Combine with friend's strategy -----
echo.
echo ============================================================
echo Stage 2: Combined backtest (CMM + friend's EW momentum)
echo ============================================================
echo Reading CMM data from: data\results%CMM_DATA_SUFFIX%\hml_returns_vw\
echo.

echo Stage 2a: results-level blend (v2)
python "C:\Projects\Momentum trading\combined\combine_strategies_v2.py"
if errorlevel 1 (
    echo [BLOCKER] Combine v2 failed.
    exit /b 1
)

echo.
echo Stage 2b: unified joint backtester (live EW momentum + CMM HML)
python "C:\Projects\Momentum trading\combined\joint_backtest.py"
if errorlevel 1 (
    echo [BLOCKER] Joint backtester failed.
    exit /b 1
)

echo.
echo ============================================================
echo Pipeline complete.
echo ============================================================
echo Outputs:
echo   data\results%CMM_DATA_SUFFIX%\           Per-window CMM results
echo   combined\combined_v2_headline.csv        v2 (return-blend) headline
echo   combined\combined_v2_equity.png          v2 equity curves
echo   combined\joint_backtest_headline.csv     Joint backtester headline
echo   combined\joint_backtest_detail.csv       Joint backtester full month-by-month
echo   combined\joint_backtest_equity.png       Joint backtester equity + regime timeline
echo   combined\joint_backtest_turnover.png     Joint backtester turnover decomposition

endlocal
