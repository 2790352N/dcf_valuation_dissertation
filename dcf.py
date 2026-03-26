from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


MARGINAL_TAX_RATE_US_POST_TCJA = 0.21
MARGINAL_TAX_RATE_US_PRE_TCJA = 0.35
TCJA_EFFECTIVE_YEAR = 2018  # TCJA enacted Dec 2017, effective for tax years from 2018
EFFECTIVE_TAX_FLOOR = 0.12
TERMINAL_GROWTH_FLOOR = 0.02
TERMINAL_GROWTH_CAP = 0.04

# Minimum spread between WACC and g to prevent near-zero or negative
# Gordon growth denominators. 1% is conservative — produces a maximum
# TV multiplier of 100x terminal UFCF.
MIN_WACC_G_SPREAD = 0.01

# Absolute WACC floor. A nominal WACC below 5% implies a cost of equity
# near or below the risk-free rate for most capital structures, which is
# economically implausible. Sub-5% WACCs arise primarily from beta
# estimation noise (very low or negative betas from short trading
# histories or low-correlation periods) compounded by the low-rate
# environment of 2020-2022.
WACC_ABSOLUTE_FLOOR = 0.05


def compute_terminal_growth(rf: float) -> float:
    return max(TERMINAL_GROWTH_FLOOR, min(rf, TERMINAL_GROWTH_CAP))


@dataclass
class DCFInputs:
    forecast_years: int = 10
    wacc: float = 0.09
    terminal_growth: float = 0.025

    revenue_cagr: Optional[float] = None
    ebit_margin: Optional[float] = None
    tax_rate: Optional[float] = None
    da_pct_rev: Optional[float] = None
    capex_pct_rev: Optional[float] = None
    dnwc_pct_rev: Optional[float] = None

    net_debt: Optional[float] = None
    shares_outstanding: Optional[float] = None
    base_year: Optional[int] = None
    history_years_for_defaults: int = 5


def _clip_rate(x: float, lo: float = -0.5, hi: float = 0.5) -> float:
    return float(np.clip(x, lo, hi))


def _safe_median(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.median())


def derive_defaults_from_history(df_hist: pd.DataFrame, n_years: int, base_year: Optional[int] = None) -> Dict[str, float]:
    df = df_hist.copy().sort_values("date").tail(n_years)

    rev = pd.to_numeric(df["revenue"], errors="coerce").dropna()
    revenue_cagr = None
    if len(rev) >= 2 and rev.iloc[0] > 0 and rev.iloc[-1] > 0:
        years = len(rev) - 1
        revenue_cagr = (rev.iloc[-1] / rev.iloc[0]) ** (1 / years) - 1

    ebit_margin = None
    if "ebit" in df.columns:
        ebit = pd.to_numeric(df["ebit"], errors="coerce")
        if not rev.empty:
            m = (ebit / rev).replace([np.inf, -np.inf], np.nan).dropna()
            ebit_margin = float(m.median()) if not m.empty else None

    tr = pd.to_numeric(df.get("tax_rate", pd.Series(dtype=float)), errors="coerce")
    tr = tr[(tr > 0) & (tr < 0.6)].dropna()
    if not tr.empty:
        tax_rate = float(tr.median())
    else:
        # Use prevailing marginal rate as fallback
        if base_year is not None and base_year < TCJA_EFFECTIVE_YEAR:
            tax_rate = MARGINAL_TAX_RATE_US_PRE_TCJA
        else:
            tax_rate = MARGINAL_TAX_RATE_US_POST_TCJA
    tax_rate = max(EFFECTIVE_TAX_FLOOR, min(0.35, tax_rate))

    da_pct = None
    da = pd.to_numeric(df.get("da", pd.Series(dtype=float)), errors="coerce")
    if not rev.empty:
        m = (da / rev).replace([np.inf, -np.inf], np.nan).dropna()
        da_pct = float(m.median()) if not m.empty else None

    capex_pct = None
    capex = pd.to_numeric(df.get("capex_spend", pd.Series(dtype=float)), errors="coerce")
    if not rev.empty:
        m = (capex / rev).replace([np.inf, -np.inf], np.nan).dropna()
        capex_pct = float(m.median()) if not m.empty else None

    dnwc_pct = None
    dnwc = pd.to_numeric(df.get("dnwc", pd.Series(dtype=float)), errors="coerce")
    if not rev.empty:
        m = (dnwc / rev).replace([np.inf, -np.inf], np.nan).dropna()
        dnwc_pct = float(m.median()) if not m.empty else None

    out = {
        "revenue_cagr": _clip_rate(revenue_cagr if revenue_cagr is not None else 0.04, -0.2, 0.3),
        "ebit_margin": float(np.clip(ebit_margin if ebit_margin is not None else 0.15, 0.02, 0.6)),
        "tax_rate": tax_rate,
        "da_pct_rev": float(np.clip(da_pct if da_pct is not None else 0.03, 0.0, 0.15)),
        "capex_pct_rev": float(np.clip(capex_pct if capex_pct is not None else 0.04, 0.0, 0.2)),
        "dnwc_pct_rev": float(np.clip(dnwc_pct if dnwc_pct is not None else 0.00, -0.1, 0.1)),
    }
    return out


def run_dcf(drivers_df: pd.DataFrame, inputs: DCFInputs) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    df = drivers_df.copy().sort_values("date").reset_index(drop=True)

    for col in ["revenue", "ebit", "tax_rate", "da", "capex_spend", "dnwc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    last = df.dropna(subset=["revenue"]).iloc[-1]
    base_revenue = float(last["revenue"])

    defaults = derive_defaults_from_history(df, inputs.history_years_for_defaults, base_year=inputs.base_year)

    rev_cagr = inputs.revenue_cagr if inputs.revenue_cagr is not None else defaults["revenue_cagr"]
    ebit_margin = inputs.ebit_margin if inputs.ebit_margin is not None else defaults["ebit_margin"]
    da_pct = inputs.da_pct_rev if inputs.da_pct_rev is not None else defaults["da_pct_rev"]
    capex_pct = inputs.capex_pct_rev if inputs.capex_pct_rev is not None else defaults["capex_pct_rev"]
    dnwc_pct = inputs.dnwc_pct_rev if inputs.dnwc_pct_rev is not None else defaults["dnwc_pct_rev"]

    effective_tax = inputs.tax_rate if inputs.tax_rate is not None else defaults["tax_rate"]
    effective_tax = max(EFFECTIVE_TAX_FLOOR, effective_tax)

    n = inputs.forecast_years
    g = inputs.terminal_growth
    wacc = inputs.wacc

    # Floor WACC at absolute minimum (plausibility bound)
    wacc_floored = False
    if wacc < WACC_ABSOLUTE_FLOOR:
        wacc = WACC_ABSOLUTE_FLOOR
        wacc_floored = True

    # Floor WACC so it's always at least g + MIN_WACC_G_SPREAD
    # This prevents undefined or extreme terminal values
    if wacc <= g + MIN_WACC_G_SPREAD:
        wacc = g + MIN_WACC_G_SPREAD
        wacc_floored = True

    years = np.arange(1, n + 1)

    # Revenue growth fade
    growth_rates = np.linspace(rev_cagr, g, n)

    revenue = np.empty(n, dtype=float)
    revenue[0] = base_revenue * (1.0 + growth_rates[0])
    for i in range(1, n):
        revenue[i] = revenue[i - 1] * (1.0 + growth_rates[i])

    # Reinvestment fade
    capex_terminal_pct = da_pct
    if capex_pct < capex_terminal_pct:
        capex_pcts = np.full(n, capex_pct)
    else:
        capex_pcts = np.linspace(capex_pct, capex_terminal_pct, n)

    dnwc_terminal_pct = 0.0
    dnwc_pcts = np.linspace(dnwc_pct, dnwc_terminal_pct, n)
    da_pcts = np.full(n, da_pct)

    ebit = revenue * ebit_margin
    nopat = ebit * (1.0 - effective_tax)
    da = revenue * da_pcts
    capex = revenue * capex_pcts
    dnwc = revenue * dnwc_pcts

    ufcf = nopat + da - capex - dnwc

    disc = 1.0 / (1.0 + wacc) ** (years - 0.5)
    pv_ufcf = ufcf * disc

    # Terminal value
    # Use prevailing marginal tax rate at the valuation date to avoid look-ahead bias
    if inputs.base_year is not None and inputs.base_year < TCJA_EFFECTIVE_YEAR:
        marginal_tax_terminal = MARGINAL_TAX_RATE_US_PRE_TCJA
    else:
        marginal_tax_terminal = MARGINAL_TAX_RATE_US_POST_TCJA

    terminal_revenue = revenue[-1] * (1.0 + g)
    terminal_ebit = terminal_revenue * ebit_margin
    terminal_nopat = terminal_ebit * (1.0 - marginal_tax_terminal)
    terminal_da = terminal_revenue * da_pct
    terminal_capex = terminal_revenue * da_pct
    terminal_dnwc = 0.0
    terminal_ufcf = terminal_nopat + terminal_da - terminal_capex - terminal_dnwc

    tv = terminal_ufcf / (wacc - g)
    pv_tv = tv * disc[-1]

    ev = float(pv_ufcf.sum() + pv_tv)

    forecast_df = pd.DataFrame(
        {
            "year": years,
            "revenue": revenue,
            "ebit": ebit,
            "tax_rate": effective_tax,
            "nopat": nopat,
            "da": da,
            "capex": capex,
            "capex_pct": capex_pcts,
            "dnwc": dnwc,
            "dnwc_pct": dnwc_pcts,
            "ufcf": ufcf,
            "discount_factor": disc,
            "pv_ufcf": pv_ufcf,
            "revenue_growth": growth_rates,
        }
    )

    summary: Dict[str, Any] = {
        "forecast_years": n,
        "assumptions": {
            "revenue_cagr": rev_cagr,
            "ebit_margin": ebit_margin,
            "tax_rate_explicit": float(effective_tax),
            "tax_rate_terminal": float(marginal_tax_terminal),
            "tax_rate_method": "effective_for_explicit_marginal_for_terminal",
            "da_pct_rev": da_pct,
            "capex_pct_rev_yr1": float(capex_pcts[0]),
            "capex_pct_rev_terminal": float(capex_pcts[-1]),
            "capex_fade": "linear_to_da_pct (maintenance)",
            "dnwc_pct_rev_yr1": float(dnwc_pcts[0]),
            "dnwc_pct_rev_terminal": float(dnwc_pcts[-1]),
            "dnwc_fade": "linear_to_zero (stable WC)",
            "wacc": wacc,
            "wacc_floored": wacc_floored,
            "terminal_growth": g,
            "terminal_growth_method": "max(2%, min(Rf, 4%))",
        },
        "enterprise_value": ev,
        "terminal_value": float(tv),
        "pv_terminal_value": float(pv_tv),
        "pv_ufcf_sum": float(pv_ufcf.sum()),
    }

    if inputs.net_debt is not None:
        equity_value = ev - float(inputs.net_debt)
        summary["net_debt"] = float(inputs.net_debt)
        summary["equity_value"] = float(equity_value)

        if inputs.shares_outstanding is not None and inputs.shares_outstanding > 0:
            summary["shares_outstanding"] = float(inputs.shares_outstanding)
            summary["implied_price_per_share"] = float(equity_value / float(inputs.shares_outstanding))

    # Sensitivity grid
    wacc_grid = np.array([wacc - 0.01, wacc, wacc + 0.01])
    g_grid = np.array([g - 0.005, g, g + 0.005])

    sens = []
    for ww in wacc_grid:
        row = {"wacc": ww}
        for gg in g_grid:
            if ww <= gg:
                val = np.nan
            else:
                disc_s = 1.0 / (1.0 + ww) ** (years - 0.5)
                pv_ufcf_s = (ufcf * disc_s).sum()
                t_rev = revenue[-1] * (1.0 + gg)
                t_nopat = t_rev * ebit_margin * (1.0 - marginal_tax_terminal)
                t_da = t_rev * da_pct
                t_capex = t_rev * da_pct
                t_ufcf = t_nopat + t_da - t_capex
                tv_s = t_ufcf / (ww - gg)
                pv_tv_s = tv_s * disc_s[-1]
                val = pv_ufcf_s + pv_tv_s
            row[f"g={gg:.3f}"] = val
        sens.append(row)

    sens_df = pd.DataFrame(sens)

    return forecast_df, summary, sens_df
