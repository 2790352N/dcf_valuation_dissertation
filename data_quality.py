"""
data_quality.py — Comprehensive data-quality and valuation-plausibility flags.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def _flag(severity: str, code: str, message: str) -> Dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def check_drivers(drivers_df: pd.DataFrame, ticker: str) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    df = drivers_df.copy()
    n_rows = len(df)

    if n_rows < 3:
        flags.append(_flag("error", "SHORT_HISTORY",
                           f"{ticker}: Only {n_rows} years of history. Minimum 3 needed, 5 preferred."))
    elif n_rows < 5:
        flags.append(_flag("warning", "SHORT_HISTORY",
                           f"{ticker}: Only {n_rows} years of history (5 preferred)."))

    key_cols = {
        "revenue": "Revenue", "ebit": "EBIT", "da": "D&A",
        "capex_spend": "CapEx", "dnwc": "ΔNWC", "tax_rate": "Tax rate",
        "interest_expense": "Interest expense",
    }
    for col, label in key_cols.items():
        if col not in df.columns:
            flags.append(_flag("warning", f"MISSING_COL_{col.upper()}",
                               f"{ticker}: Column '{col}' not present in drivers."))
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(series.isna().sum())
        if n_missing == n_rows:
            flags.append(_flag("error", f"ALL_MISSING_{col.upper()}",
                               f"{ticker}: {label} missing for ALL {n_rows} years."))
        elif n_missing > 0:
            missing_dates = list(df.loc[series.isna(), "date"].values)
            flags.append(_flag("warning", f"PARTIAL_MISSING_{col.upper()}",
                               f"{ticker}: {label} missing for {n_missing}/{n_rows} years: {missing_dates}"))

    if "tax_rate" in df.columns:
        tr = pd.to_numeric(df["tax_rate"], errors="coerce").dropna()
        valid_tr = tr[(tr > 0) & (tr < 0.6)]
        if len(valid_tr) == 0 and len(tr) > 0:
            flags.append(_flag("warning", "TAX_RATE_ALL_INVALID",
                               f"{ticker}: No valid tax rates. Fallback 21% marginal used."))
        elif len(valid_tr) < len(tr):
            n_excluded = len(tr) - len(valid_tr)
            flags.append(_flag("info", "TAX_RATE_SOME_EXCLUDED",
                               f"{ticker}: {n_excluded} year(s) excluded from tax rate estimate."))
    return flags


def check_wacc_inputs(
    ticker: str, beta: Optional[float], market_cap: Optional[float],
    total_debt: Optional[float], interest_expense: Optional[float],
    wacc_info: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []

    if beta is None:
        flags.append(_flag("error", "MISSING_BETA",
                           f"{ticker}: Beta not available. WACC falls back to 9% default."))
    elif beta < 0:
        flags.append(_flag("warning", "NEGATIVE_BETA",
                           f"{ticker}: Beta is {beta:.3f} (negative)."))
    elif beta > 3.0:
        flags.append(_flag("warning", "EXTREME_BETA",
                           f"{ticker}: Beta is {beta:.3f} (>3.0)."))

    if market_cap is None:
        flags.append(_flag("error", "MISSING_MARKET_CAP",
                           f"{ticker}: Market cap not available. WACC falls back to 9%."))
    if total_debt is None:
        flags.append(_flag("error", "MISSING_TOTAL_DEBT",
                           f"{ticker}: Total debt not available. WACC falls back to 9%."))

    if interest_expense is None and total_debt is not None and total_debt > 0:
        flags.append(_flag("warning", "MISSING_INTEREST_EXPENSE",
                           f"{ticker}: Interest expense not available."))

    if wacc_info is None:
        flags.append(_flag("error", "WACC_FALLBACK",
                           f"{ticker}: WACC could not be computed. Using 9% fallback."))
    else:
        wacc_val = wacc_info.get("wacc", 0)
        if wacc_val < 0.04:
            flags.append(_flag("warning", "WACC_LOW", f"{ticker}: WACC is {wacc_val:.4f} (<4%)."))
        elif wacc_val > 0.20:
            flags.append(_flag("warning", "WACC_HIGH", f"{ticker}: WACC is {wacc_val:.4f} (>20%)."))

        rd_method = wacc_info.get("cost_of_debt_method", "")
        if "fallback" in str(rd_method).lower():
            flags.append(_flag("warning", "COST_OF_DEBT_FALLBACK",
                               f"{ticker}: Cost of debt used fallback ({rd_method})."))
    return flags


def check_bridge(ticker: str, bridge: Dict[str, Optional[float]], bs_date: Optional[str]) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    if bs_date is None:
        flags.append(_flag("error", "MISSING_BALANCE_SHEET", f"{ticker}: No balance sheet for as-of year."))
        return flags
    if bridge.get("cash") is None:
        flags.append(_flag("warning", "MISSING_CASH", f"{ticker}: Cash not found ({bs_date})."))
    if bridge.get("total_debt") is None:
        flags.append(_flag("warning", "MISSING_DEBT", f"{ticker}: Total debt not found ({bs_date})."))
    if bridge.get("net_debt") is None:
        flags.append(_flag("error", "MISSING_NET_DEBT", f"{ticker}: Net debt could not be computed."))
    return flags


def check_derived_assumptions(ticker: str, assumptions: Dict[str, Any]) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []

    rev_cagr = assumptions.get("revenue_cagr", 0)
    if rev_cagr >= 0.30:
        flags.append(_flag("warning", "REV_CAGR_AT_CAP", f"{ticker}: Revenue CAGR hit 30% ceiling."))
    elif rev_cagr <= -0.20:
        flags.append(_flag("warning", "REV_CAGR_AT_FLOOR", f"{ticker}: Revenue CAGR hit -20% floor."))
    elif rev_cagr < 0:
        flags.append(_flag("info", "REV_CAGR_NEGATIVE", f"{ticker}: Revenue CAGR is {rev_cagr:.2%}."))

    ebit_m = assumptions.get("ebit_margin", 0)
    if ebit_m <= 0.02:
        flags.append(_flag("warning", "EBIT_MARGIN_AT_FLOOR", f"{ticker}: EBIT margin hit 2% floor."))
    elif ebit_m >= 0.60:
        flags.append(_flag("warning", "EBIT_MARGIN_AT_CAP", f"{ticker}: EBIT margin hit 60% ceiling."))
    elif ebit_m < 0.05:
        flags.append(_flag("info", "EBIT_MARGIN_LOW", f"{ticker}: EBIT margin is {ebit_m:.2%}."))

    tax_explicit = assumptions.get("tax_rate_explicit", 0.21)
    if tax_explicit <= 0.12:
        flags.append(_flag("warning", "TAX_RATE_AT_FLOOR",
                           f"{ticker}: Effective tax rate hit 12% floor."))

    capex_pct = assumptions.get("capex_pct_rev", 0)
    da_pct = assumptions.get("da_pct_rev", 0)
    if capex_pct > 0 and da_pct > 0 and capex_pct > da_pct * 3:
        flags.append(_flag("info", "CAPEX_MUCH_ABOVE_DA",
                           f"{ticker}: CapEx/Rev ({capex_pct:.2%}) >3x D&A/Rev ({da_pct:.2%})."))
    return flags


def check_valuation_output(ticker: str, summary: Dict[str, Any]) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    ev = summary.get("enterprise_value")
    pv_tv = summary.get("pv_terminal_value")

    if ev is not None and ev <= 0:
        flags.append(_flag("error", "NEGATIVE_EV", f"{ticker}: EV is {ev:,.0f} (non-positive)."))
    if ev is not None and pv_tv is not None and ev > 0:
        tv_pct = pv_tv / ev
        if tv_pct > 0.95:
            flags.append(_flag("warning", "TV_DOMINANCE_EXTREME", f"{ticker}: TV is {tv_pct:.1%} of EV."))
        elif tv_pct > 0.85:
            flags.append(_flag("info", "TV_DOMINANCE_HIGH", f"{ticker}: TV is {tv_pct:.1%} of EV."))

    equity = summary.get("equity_value")
    if equity is not None and equity < 0:
        flags.append(_flag("warning", "NEGATIVE_EQUITY", f"{ticker}: Equity is {equity:,.0f} (negative)."))
    price = summary.get("implied_price_per_share")
    if price is not None and price < 0:
        flags.append(_flag("warning", "NEGATIVE_IMPLIED_PRICE", f"{ticker}: Price is {price:.2f} (negative)."))
    return flags


def run_all_checks(
    ticker: str, drivers_df: pd.DataFrame,
    beta: Optional[float], market_cap: Optional[float],
    total_debt: Optional[float], interest_expense: Optional[float],
    wacc_info: Optional[Dict[str, Any]], bridge: Dict[str, Optional[float]],
    bs_date: Optional[str], assumptions: Dict[str, Any], summary: Dict[str, Any],
) -> List[Dict[str, str]]:
    all_flags: List[Dict[str, str]] = []
    all_flags.extend(check_drivers(drivers_df, ticker))
    all_flags.extend(check_wacc_inputs(ticker, beta, market_cap, total_debt, interest_expense, wacc_info))
    all_flags.extend(check_bridge(ticker, bridge, bs_date))
    all_flags.extend(check_derived_assumptions(ticker, assumptions))
    all_flags.extend(check_valuation_output(ticker, summary))
    severity_order = {"error": 0, "warning": 1, "info": 2}
    all_flags.sort(key=lambda f: severity_order.get(f["severity"], 3))
    return all_flags


def summarise_flags(flags: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for f in flags:
        counts[f.get("severity", "info")] = counts.get(f.get("severity", "info"), 0) + 1
    return counts
