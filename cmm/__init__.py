"""Characteristic-Managed Momentum (CMM) model."""

from cmm.data import prepare_cmm_data, cross_sectional_normalize
from cmm.fetch_data import fetch_sp500_cmm_data
from cmm.model import CMMModel
from cmm.training import run_expanding_window
from cmm.portfolio import build_hml_portfolio, get_nyse_tickers, hml_summary

# Optional: requires the `wrds` package + a WRDS account.
try:
    from cmm.fetch_data_jkp import fetch_jkp_cmm_data  # noqa: F401
except ImportError:
    fetch_jkp_cmm_data = None  # type: ignore

try:
    from cmm.ffn import CMMFFN, CMMFFNWrapper
    __all__ = [
        "CMMModel",
        "CMMFFN",
        "CMMFFNWrapper",
        "prepare_cmm_data",
        "cross_sectional_normalize",
        "fetch_sp500_cmm_data",
        "run_expanding_window",
        "build_hml_portfolio",
        "get_nyse_tickers",
        "hml_summary",
    ]
except ImportError:
    __all__ = [
        "CMMModel",
        "prepare_cmm_data",
        "cross_sectional_normalize",
        "fetch_sp500_cmm_data",
        "run_expanding_window",
        "build_hml_portfolio",
        "get_nyse_tickers",
        "hml_summary",
    ]
