# src/main.py
from __future__ import annotations

import json
import traceback
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from data_loader import EODHDConfig, EODHDFundamentalsClient
from financials import parse_eodhd_fundamentals
from dcf_inputs import build_historical_drivers
from balance_sheet import extract_cash_and_debt
from wacc import WACCInputs, compute_wacc
from dcf import DCFInputs, compute_terminal_growth, run_dcf
from price_loader import EODHDPriceClient, EODHDPriceConfig
from beta import compute_regression_beta
from historical_mcap import HistoricalMcapClient
from macro_loader import MacroSeriesLoader
from cross_sectional import compute_all_cross_sectional
from data_quality import run_all_checks, summarise_flags


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS = PROJECT_ROOT / "data" / "results"
MACRO_CSV = PROJECT_ROOT / "data" / "macro_series.csv"


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)


def load_tickers_with_asof(path: Path, default_asof_year: int = 2026) -> List[Tuple[str, int]]:
    if not path.exists():
        return []
    out: List[Tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.replace("\t", ",").split(",") if p.strip()]
        ticker = parts[0]
        asof_year = default_asof_year
        if len(parts) >= 2:
            try:
                asof_year = int(parts[1])
            except ValueError:
                asof_year = default_asof_year
        if asof_year not in range(2008, 2027):
            asof_year = default_asof_year
        out.append((ticker, asof_year))
    return out


def filter_drivers_asof(drivers_df: pd.DataFrame, asof_year: int) -> pd.DataFrame:
    df = drivers_df.copy()
    if "date" not in df.columns or df.empty:
        return pd.DataFrame()
    dt = pd.to_datetime(df["date"], errors="coerce")
    df = df[dt.notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["__year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df = df[df["__year"] <= asof_year].sort_values("date").reset_index(drop=True)
    df = df.drop(columns=["__year"], errors="ignore")
    return df


def pick_balance_sheet_asof(fin, asof_year: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not fin.balance_annual or not fin.balance_annual.items_by_date:
        return None, None
    candidates: List[Tuple[int, str]] = []
    for d in fin.balance_annual.items_by_date.keys():
        try:
            y = int(pd.to_datetime(d).year)
        except Exception:
            continue
        if y <= asof_year:
            candidates.append((y, d))
    if not candidates:
        return None, None
    _, best_date = sorted(candidates, key=lambda x: (x[0], x[1]))[-1]
    return best_date, fin.balance_annual.items_by_date[best_date]


def save_parsed_snapshot(ticker: str, fin, asof_year: int) -> None:
    income_dates = fin.income_annual.dates_desc()
    cf_dates = fin.cashflow_annual.dates_desc()
    snapshot = {
        "ticker": fin.ticker, "asof_year": asof_year,
        "currency": fin.currency, "shares_outstanding": fin.shares_outstanding,
        "market_cap_current": fin.market_cap, "beta_vendor": fin.beta,
        "income_annual_dates": income_dates, "cashflow_annual_dates": cf_dates,
    }
    out_path = DATA_PROCESSED / f"{ticker.replace('/', '_')}_parsed_snapshot_asof_{asof_year}.json"
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def run_one(
    ticker: str,
    asof_year: int,
    fundamentals_client: EODHDFundamentalsClient,
    price_client: EODHDPriceClient,
    mcap_client: HistoricalMcapClient,
    macro: MacroSeriesLoader,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    # 1) Fetch fundamentals
    raw = fundamentals_client.get_fundamentals(ticker=ticker, use_cache=True, force_refresh=force_refresh)

    # 2) Parse
    fin = parse_eodhd_fundamentals(ticker, raw)
    save_parsed_snapshot(ticker, fin, asof_year)

    # 3) Build drivers — with early validation
    drivers_df, dq = build_historical_drivers(fin, min_years=5)

    # Check for empty DataFrame before proceeding
    if drivers_df.empty or "date" not in drivers_df.columns:
        raise RuntimeError(f"No financial statement data available for {ticker}")

    # Check that date column has valid values
    valid_dates = pd.to_datetime(drivers_df["date"], errors="coerce").dropna()
    if valid_dates.empty:
        raise RuntimeError(f"No valid fiscal year dates found for {ticker}")

    numeric_cols = [
        "revenue", "ebit", "pretax_income", "tax_expense", "tax_rate",
        "interest_expense", "cfo", "da", "capex_spend", "dnwc", "nopat",
        "ufcf", "fcf_reported", "ufcf_minus_reported_fcf"
    ]
    for c in numeric_cols:
        if c in drivers_df.columns:
            drivers_df[c] = pd.to_numeric(drivers_df[c], errors="coerce")

    if dq.missing_fields:
        raise RuntimeError(f"Missing required driver fields: {dq.missing_fields}")

    # 4) As-of filter
    drivers_asof = filter_drivers_asof(drivers_df, asof_year)
    if drivers_asof.empty:
        raise RuntimeError(f"No driver rows available with date-year <= asof_year={asof_year}")

    drivers_df = drivers_asof
    base_date_used = str(drivers_df.iloc[-1]["date"])
    base_year_used = int(pd.to_datetime(base_date_used).year)

    drivers_path = DATA_PROCESSED / f"{ticker.replace('/', '_')}_drivers_asof_{asof_year}.csv"
    drivers_df.to_csv(drivers_path, index=False)

    # 5) Balance sheet
    bs_date, bs_items = pick_balance_sheet_asof(fin, asof_year)
    bridge = {"cash": None, "total_debt": None, "net_debt": None}
    if bs_items:
        bridge = extract_cash_and_debt(bs_items)

    # 6) Market cap as-of
    mcap_asof: Optional[float] = None
    mcap_date_used: Optional[str] = None
    mcap_source: str = "none"
    close_asof: Optional[float] = None
    price_date_used: Optional[str] = None

    hist_mcap_date, hist_mcap_val = mcap_client.get_mcap_near_date(
        ticker=ticker, target_date=base_date_used, window_days=30,
        use_cache=True, force_refresh=force_refresh,
    )
    if hist_mcap_val is not None:
        mcap_asof = hist_mcap_val
        mcap_date_used = hist_mcap_date
        mcap_source = "historical_mcap_api"

    if mcap_asof is None and fin.shares_outstanding:
        price_date_used, close_asof = price_client.get_close_near_date(
            ticker=ticker, target_date=base_date_used, window_days=7,
            use_cache=True, force_refresh=False,
        )
        if close_asof is not None:
            mcap_asof = float(close_asof) * float(fin.shares_outstanding)
            mcap_date_used = price_date_used
            mcap_source = "close_x_shares"

    if mcap_asof is None and fin.market_cap:
        mcap_asof = fin.market_cap
        mcap_source = "current_fundamentals"

    # 7) Beta
    computed_beta, beta_info = compute_regression_beta(
        price_client=price_client, ticker=ticker, as_of_date=base_date_used,
        lookback_years=5, use_cache=True, force_refresh=False,
    )
    beta_used = computed_beta if computed_beta is not None else fin.beta
    beta_source = "computed_5yr_monthly_ols_vasicek" if computed_beta is not None else "vendor_eodhd"

    # 8) Macro inputs
    rf, erp, macro_matched_date = macro.get_rf_erp(base_date_used)

    # 9) Terminal growth from Rf
    terminal_g = compute_terminal_growth(rf)

    # Extract base-year inputs for WACC
    base_row = drivers_df.iloc[-1]

    interest_expense_base = None
    if "interest_expense" in base_row.index:
        ie = base_row["interest_expense"]
        if pd.notna(ie):
            interest_expense_base = float(ie)

    ebit_base = None
    if "ebit" in base_row.index:
        eb = base_row["ebit"]
        if pd.notna(eb):
            ebit_base = float(eb)

    # 10) WACC
    wacc_info = None
    mcap_for_wacc = mcap_asof

    # Use prevailing marginal tax rate at valuation date to avoid look-ahead bias
    if base_year_used < 2018:
        marginal_tax_for_wacc = 0.35  # Pre-TCJA federal corporate rate
    else:
        marginal_tax_for_wacc = 0.21  # Post-TCJA (Tax Cuts and Jobs Act, 2017)

    if mcap_for_wacc and bridge["total_debt"] and beta_used:
        wacc_info = compute_wacc(
            market_cap=float(mcap_for_wacc),
            total_debt=float(bridge["total_debt"]),
            beta=float(beta_used),
            wacc_inputs=WACCInputs(
                risk_free_rate=rf,
                equity_risk_premium=erp,
                marginal_tax_rate=marginal_tax_for_wacc,
                cost_of_debt=None,
            ),
            effective_tax_rate=None,
            interest_expense=interest_expense_base,
            ebit=ebit_base,
        )

    # 11) DCF
    dcf_inputs = DCFInputs(
        forecast_years=10,
        wacc=(wacc_info["wacc"] if wacc_info else 0.09),
        terminal_growth=terminal_g,
        net_debt=bridge["net_debt"],
        shares_outstanding=fin.shares_outstanding,
        base_year=base_year_used,
    )

    forecast_df, summary, sens_df = run_dcf(drivers_df, dcf_inputs)

    # 12) Cross-sectional variables
    base_revenue = None
    rev_col = pd.to_numeric(drivers_df["revenue"], errors="coerce").dropna()
    if not rev_col.empty:
        base_revenue = float(rev_col.iloc[-1])

    tv_pct_of_ev = None
    ev_val = summary.get("enterprise_value")
    pv_tv_val = summary.get("pv_terminal_value")
    if ev_val and ev_val > 0 and pv_tv_val:
        tv_pct_of_ev = pv_tv_val / ev_val

    cross_vars = compute_all_cross_sectional(
        raw_fundamentals=raw,
        drivers_df=drivers_df,
        total_debt=bridge.get("total_debt"),
        mcap_asof=mcap_asof,
        balance_items=bs_items,
        base_revenue=base_revenue,
        ebit_margin=summary["assumptions"]["ebit_margin"],
        tv_pct_of_ev=tv_pct_of_ev,
    )

    # 13) Data quality flags
    flags = run_all_checks(
        ticker=ticker, drivers_df=drivers_df, beta=beta_used,
        market_cap=mcap_for_wacc, total_debt=bridge.get("total_debt"),
        interest_expense=interest_expense_base, wacc_info=wacc_info,
        bridge=bridge, bs_date=bs_date, assumptions=summary["assumptions"],
        summary=summary,
    )
    flag_counts = summarise_flags(flags)

    if flags:
        for f in flags:
            sev = f["severity"].upper()
            print(f"  [{sev}] {f['code']}: {f['message']}")

    # 14) Save outputs
    forecast_path = DATA_RESULTS / f"{ticker.replace('/', '_')}_dcf_forecast_asof_{asof_year}.csv"
    sens_path = DATA_RESULTS / f"{ticker.replace('/', '_')}_dcf_sensitivity_asof_{asof_year}.csv"
    summary_path = DATA_RESULTS / f"{ticker.replace('/', '_')}_dcf_summary_asof_{asof_year}.json"

    forecast_df.to_csv(forecast_path, index=False)
    sens_df.to_csv(sens_path, index=False)

    summary_out = dict(summary)
    summary_out["asof_year_requested"] = asof_year
    summary_out["base_date_used"] = base_date_used
    summary_out["base_year_used"] = base_year_used
    summary_out["drivers_file"] = str(drivers_path)

    summary_out["balance_sheet_date_used"] = bs_date
    summary_out["balance_bridge_used"] = bridge

    summary_out["beta_used"] = beta_used
    summary_out["beta_source"] = beta_source
    summary_out["beta_vendor"] = fin.beta
    summary_out["beta_computed_info"] = beta_info
    summary_out["market_cap_current"] = fin.market_cap

    summary_out["mcap_asof"] = mcap_asof
    summary_out["mcap_date_used"] = mcap_date_used
    summary_out["mcap_source"] = mcap_source
    summary_out["market_cap_used_for_wacc"] = float(mcap_for_wacc) if mcap_for_wacc is not None else None

    summary_out["macro_inputs"] = {"rf": rf, "erp": erp, "g": terminal_g, "matched_date": macro_matched_date}
    summary_out["cross_sectional"] = cross_vars

    if wacc_info:
        summary_out["wacc_breakdown"] = wacc_info

    summary_out["data_quality_flags"] = flags
    summary_out["data_quality_counts"] = flag_counts

    summary_path.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    # 15) Consolidated result row
    result: Dict[str, Any] = {
        "ticker": ticker, "asof_year": asof_year, "base_date_used": base_date_used,
        "currency": fin.currency,

        "rf_used": rf, "erp_used": erp, "terminal_g_used": terminal_g,
        "macro_matched_date": macro_matched_date,

        "beta_used": beta_used, "beta_source": beta_source,
        "beta_vendor": fin.beta, "beta_r_squared": beta_info.get("r_squared"),
        "beta_raw": beta_info.get("raw_beta"),

        "shares_outstanding": fin.shares_outstanding,

        "mcap_asof": mcap_asof, "mcap_date_used": mcap_date_used,
        "mcap_source": mcap_source, "market_cap_current": fin.market_cap,
        "market_cap_used_for_wacc": float(mcap_for_wacc) if mcap_for_wacc is not None else None,

        "balance_sheet_date_used": bs_date,
        "cash_asof": bridge["cash"], "total_debt_asof": bridge["total_debt"],
        "net_debt_asof": bridge["net_debt"],

        "forecast_years": summary.get("forecast_years"),
        "wacc_used": summary["assumptions"]["wacc"],
        "wacc_floored": summary["assumptions"].get("wacc_floored", False),
        "terminal_growth": summary["assumptions"]["terminal_growth"],
        "revenue_cagr_start": summary["assumptions"]["revenue_cagr"],
        "ebit_margin": summary["assumptions"]["ebit_margin"],
        "tax_rate_explicit": summary["assumptions"]["tax_rate_explicit"],
        "tax_rate_terminal": summary["assumptions"]["tax_rate_terminal"],

        "ev": summary.get("enterprise_value"),
        "equity_value": summary.get("equity_value"),
        "implied_price_per_share": summary.get("implied_price_per_share"),

        "pv_ufcf_sum": summary.get("pv_ufcf_sum"),
        "pv_terminal_value": summary.get("pv_terminal_value"),

        "drivers_file": str(drivers_path),
        "forecast_file": str(forecast_path),
        "summary_file": str(summary_path),

        "dq_errors": flag_counts.get("error", 0),
        "dq_warnings": flag_counts.get("warning", 0),
        "dq_infos": flag_counts.get("info", 0),
    }

    if wacc_info:
        result["cost_of_equity"] = wacc_info.get("cost_of_equity")
        result["cost_of_debt"] = wacc_info.get("cost_of_debt")
        result["cost_of_debt_method"] = wacc_info.get("cost_of_debt_method")
        result["weight_equity"] = wacc_info.get("weights", {}).get("equity")
        result["weight_debt"] = wacc_info.get("weights", {}).get("debt")

    result.update(cross_vars)

    return result


def main() -> None:
    load_dotenv()
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise RuntimeError("Missing EODHD_API_KEY. Add it to your .env file.")

    ensure_dirs()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("EODHD_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("Missing EODHD_API_KEY.")

    macro = MacroSeriesLoader(str(MACRO_CSV))
    print(macro.summary())

    cfg = EODHDConfig(api_key=api_key)
    fundamentals_client = EODHDFundamentalsClient(cfg, cache_dir=str(DATA_RAW))
    price_client = EODHDPriceClient(EODHDPriceConfig(api_key=api_key, cache_dir=str(DATA_RAW)))
    mcap_client = HistoricalMcapClient(api_key=api_key, cache_dir=str(DATA_RAW))

    items = load_tickers_with_asof(PROJECT_ROOT / "tickers.txt", default_asof_year=2026)
    if not items:
        items = [("AAPL.US", 2026), ("TSLA.US", 2026), ("AMZN.US", 2026)]

    print(f"\nLoaded {len(items)} tickers from tickers.txt")

    force_refresh = False
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for i, (t, asof_year) in enumerate(items, start=1):
        print(f"\n[{i}/{len(items)}] Running {t} as-of {asof_year}...")
        try:
            row = run_one(t, asof_year, fundamentals_client, price_client,
                          mcap_client, macro, force_refresh)
            results.append(row)
            ev = row.get("ev")
            eq = row.get("equity_value")
            if ev is not None:
                msg = f"[OK] {t} as-of {asof_year} EV={ev:,.0f}"
                if eq is not None:
                    msg += f" Equity={eq:,.0f}"
                print(msg)
            else:
                print(f"[OK] {t} as-of {asof_year} (no EV?)")

            dq_e = row.get("dq_errors", 0)
            dq_w = row.get("dq_warnings", 0)
            if dq_e > 0 or dq_w > 0:
                print(f"  Data quality: {dq_e} error(s), {dq_w} warning(s)")

        except Exception as e:
            tb = traceback.format_exc()
            errors.append({
                "ticker": t, "asof_year": asof_year,
                "error": str(e),
                "traceback_head": "\n".join(tb.splitlines()[:60]),
            })
            print(f"[FAIL] {t} as-of {asof_year}: {e}")

    if results:
        df_res = pd.DataFrame(results).sort_values(["ticker", "asof_year"])
        out_path = DATA_RESULTS / "dcf_results.csv"
        df_res.to_csv(out_path, index=False)
        print(f"\nWrote consolidated results -> {out_path}")
        total_errors = df_res["dq_errors"].sum()
        total_warnings = df_res["dq_warnings"].sum()
        clean = len(df_res[df_res["dq_errors"] == 0])
        print(f"Data quality summary: {clean}/{len(df_res)} tickers clean (0 errors), "
              f"{int(total_errors)} total errors, {int(total_warnings)} total warnings")

    if errors:
        df_err = pd.DataFrame(errors).sort_values(["ticker", "asof_year"])
        err_path = DATA_RESULTS / "dcf_errors.csv"
        try:
            df_err.to_csv(err_path, index=False)
            print(f"Wrote errors -> {err_path}")
        except PermissionError:
            alt = DATA_RESULTS / f"dcf_errors_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_err.to_csv(alt, index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
