"""
beta.py — Compute regression beta from monthly returns.

Methodology: 5-year rolling OLS of monthly simple (holding-period) returns
against a market index (S&P 500), followed by Vasicek (Bloomberg) adjustment.

Raw beta: Gebhardt, Lee & Swaminathan (2001) — "compute betas using
five-year rolling regressions of monthly returns on a market index."
Simple returns are used to match the CRSP holding-period return convention
employed by Gebhardt et al.

Adjusted beta: Bloomberg convention — 0.67 × raw_beta + 0.33 × 1.0
This approximates the Bayesian shrinkage estimator of Vasicek (1973),
shrinking extreme betas toward the market average of 1.0. The adjustment
reflects the empirical tendency of betas to mean-revert (Blume, 1971),
making the adjusted beta more appropriate for a 10-year DCF horizon.

Source (raw): Gebhardt, W.R., Lee, C.M.C. and Swaminathan, B. (2001)
    'Toward an implied cost of capital', JAR, 39(1), pp. 135-176.
Source (adjustment): Vasicek, O.A. (1973) 'A note on using cross-sectional
    information in Bayesian estimation of security betas', Journal of
    Finance, 28(5), pp. 1233-1239.
Source (mean reversion): Blume, M.E. (1971) 'On the assessment of risk',
    Journal of Finance, 26(1), pp. 1-10.
    Bloomberg terminal uses the same 0.67/0.33 weighting.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from price_loader import EODHDPriceClient


SP500_TICKER = "GSPC.INDX"
MIN_MONTHS = 36

# Bloomberg/Vasicek adjustment weights
BETA_RAW_WEIGHT = 0.67
BETA_PRIOR_WEIGHT = 0.33
BETA_PRIOR = 1.0  # market beta


def _monthly_returns(candles: list[dict]) -> pd.Series:
    """
    Convert daily candles to monthly simple (holding-period) returns.
    Uses adjusted_close to handle splits and dividends correctly.
    Simple returns match the CRSP convention used by Gebhardt, Lee
    and Swaminathan (2001).
    """
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])

    if "adjusted_close" in df.columns:
        df["price"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    else:
        df["price"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["price"]).sort_values("date")

    monthly = df.set_index("date")["price"].resample("ME").last().dropna()
    simple_ret = (monthly / monthly.shift(1) - 1.0).dropna()
    return simple_ret


def compute_regression_beta(
    price_client: EODHDPriceClient,
    ticker: str,
    as_of_date: str,
    lookback_years: int = 5,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Tuple[Optional[float], dict]:
    """
    Compute beta via OLS regression of monthly log-returns,
    then apply Bloomberg/Vasicek adjustment.

    Returns:
        (adjusted_beta, info_dict)
        adjusted_beta is None if insufficient data.
        info_dict contains raw_beta, adjusted_beta, r_squared, etc.
    """
    dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    date_from = (dt - timedelta(days=lookback_years * 365 + 30)).isoformat()
    date_to = dt.isoformat()

    info = {
        "method": "5yr_monthly_ols_vasicek_adjusted",
        "return_type": "simple_holding_period",
        "price_field": "adjusted_close",
        "lookback_years": lookback_years,
        "as_of_date": as_of_date,
        "date_from": date_from,
        "date_to": date_to,
        "n_months": 0,
        "r_squared": None,
        "raw_beta": None,
        "adjusted_beta": None,
        "adjustment": f"{BETA_RAW_WEIGHT:.2f} × raw + {BETA_PRIOR_WEIGHT:.2f} × {BETA_PRIOR:.1f}",
        "alpha": None,
        "fallback_used": False,
    }

    # Fetch stock prices
    stock_candles = price_client.get_eod_range(
        ticker=ticker, date_from=date_from, date_to=date_to,
        use_cache=use_cache, force_refresh=force_refresh,
    )

    # Fetch market (S&P 500) prices
    market_candles = price_client.get_eod_range(
        ticker=SP500_TICKER, date_from=date_from, date_to=date_to,
        use_cache=use_cache, force_refresh=force_refresh,
    )

    if not stock_candles or not market_candles:
        info["error"] = "No price data returned for stock or market index"
        return None, info

    stock_ret = _monthly_returns(stock_candles)
    market_ret = _monthly_returns(market_candles)

    combined = pd.DataFrame({
        "stock": stock_ret,
        "market": market_ret,
    }).dropna()

    n_months = len(combined)
    info["n_months"] = n_months

    if n_months < MIN_MONTHS:
        info["error"] = f"Only {n_months} common months (need {MIN_MONTHS}+)"
        return None, info

    x = combined["market"].values
    y = combined["stock"].values

    x_mean = x.mean()
    y_mean = y.mean()

    cov_xy = ((x - x_mean) * (y - y_mean)).sum()
    var_x = ((x - x_mean) ** 2).sum()

    if var_x == 0:
        info["error"] = "Zero variance in market returns"
        return None, info

    raw_beta = float(cov_xy / var_x)
    alpha = float(y_mean - raw_beta * x_mean)

    # R-squared
    y_hat = alpha + raw_beta * x
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Vasicek / Bloomberg adjustment
    adjusted_beta = BETA_RAW_WEIGHT * raw_beta + BETA_PRIOR_WEIGHT * BETA_PRIOR

    info["raw_beta"] = raw_beta
    info["adjusted_beta"] = adjusted_beta
    info["alpha"] = alpha
    info["r_squared"] = r_squared

    # Return adjusted beta as the primary value
    return adjusted_beta, info
