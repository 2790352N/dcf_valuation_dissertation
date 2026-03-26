"""
cross_sectional.py — Compute cross-sectional variables for regression analysis.

Extracts sector/industry from EODHD fundamentals and computes firm-level
metrics needed for H1/H2 hypothesis testing:
  - Sector and GIC classification
  - Leverage (Debt/EV using market values)
  - Cash flow volatility (CV of historical UFCF)
  - Size (market cap, revenue)
  - Profitability (EBIT margin)
  - Terminal value share of EV
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def extract_classifications(raw_fundamentals: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Extract sector and industry classifications from EODHD fundamentals JSON.
    """
    g = raw_fundamentals.get("General", {})
    return {
        "sector": g.get("Sector"),
        "industry": g.get("Industry"),
        "gic_sector": g.get("GicSector"),
        "gic_group": g.get("GicGroup"),
        "gic_industry": g.get("GicIndustry"),
        "gic_sub_industry": g.get("GicSubIndustry"),
    }


def compute_leverage(
    total_debt: Optional[float],
    mcap_asof: Optional[float],
    book_equity: Optional[float],
) -> Dict[str, Optional[float]]:
    """
    Compute leverage ratios.

    Debt/EV (market): Total Debt / (Market Cap + Total Debt)
      - Uses market-value weights, consistent with WACC methodology.
      - Damodaran uses market cap for equity and book value of interest-bearing
        debt as debt proxy (see methodology section 3.3.3).

    Debt/Equity (book): Total Debt / Total Stockholder Equity
      - Common alternative; can be negative if book equity is negative.
    """
    debt_to_ev = None
    if total_debt is not None and mcap_asof is not None:
        ev_proxy = mcap_asof + total_debt
        if ev_proxy > 0:
            debt_to_ev = total_debt / ev_proxy

    debt_to_book_equity = None
    if total_debt is not None and book_equity is not None and book_equity != 0:
        debt_to_book_equity = total_debt / book_equity

    return {
        "debt_to_ev": debt_to_ev,
        "debt_to_book_equity": debt_to_book_equity,
    }


def compute_cf_volatility(drivers_df: pd.DataFrame, n_years: int = 5) -> Dict[str, Optional[float]]:
    """
    Compute cash flow volatility from historical drivers.

    Coefficient of variation (CV) = std(UFCF) / |mean(UFCF)| over the
    lookback window. Higher CV indicates more volatile/unpredictable
    cash flows, which should increase DCF valuation error (H2).

    Also computes EBIT CV as an alternative measure.
    """
    df = drivers_df.copy().sort_values("date").tail(n_years)

    ufcf_cv = None
    ufcf = pd.to_numeric(df.get("ufcf", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(ufcf) >= 3:
        mean_val = ufcf.mean()
        if abs(mean_val) > 0:
            ufcf_cv = float(ufcf.std() / abs(mean_val))

    ebit_cv = None
    ebit = pd.to_numeric(df.get("ebit", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(ebit) >= 3:
        mean_val = ebit.mean()
        if abs(mean_val) > 0:
            ebit_cv = float(ebit.std() / abs(mean_val))

    return {
        "ufcf_cv": ufcf_cv,
        "ebit_cv": ebit_cv,
        "n_years_used": len(ufcf),
    }


def extract_book_equity(balance_items: Dict[str, Any]) -> Optional[float]:
    """
    Extract total stockholder equity from balance sheet items.
    Uses totalStockholderEquity which is EODHD's standard field for
    total shareholders' equity (common equity including retained earnings).
    """
    val = balance_items.get("totalStockholderEquity")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    return None


def compute_all_cross_sectional(
    raw_fundamentals: Dict[str, Any],
    drivers_df: pd.DataFrame,
    total_debt: Optional[float],
    mcap_asof: Optional[float],
    balance_items: Optional[Dict[str, Any]],
    base_revenue: Optional[float],
    ebit_margin: Optional[float],
    tv_pct_of_ev: Optional[float],
) -> Dict[str, Any]:
    """
    Compute all cross-sectional variables in one call.
    Returns a flat dict suitable for adding to the results CSV.
    """
    out: Dict[str, Any] = {}

    # Classifications
    classifications = extract_classifications(raw_fundamentals)
    out.update(classifications)

    # Book equity from balance sheet
    book_equity = None
    if balance_items:
        book_equity = extract_book_equity(balance_items)

    # Leverage
    leverage = compute_leverage(total_debt, mcap_asof, book_equity)
    out["debt_to_ev"] = leverage["debt_to_ev"]
    out["debt_to_book_equity"] = leverage["debt_to_book_equity"]
    out["book_equity"] = book_equity

    # Cash flow volatility
    cf_vol = compute_cf_volatility(drivers_df, n_years=5)
    out["ufcf_cv"] = cf_vol["ufcf_cv"]
    out["ebit_cv"] = cf_vol["ebit_cv"]

    # Size
    out["mcap_asof_size"] = mcap_asof
    out["base_revenue"] = base_revenue

    # Profitability
    out["ebit_margin_used"] = ebit_margin

    # Terminal value share
    out["tv_pct_of_ev"] = tv_pct_of_ev

    return out
