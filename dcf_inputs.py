from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from financials import Financials


@dataclass
class DataQuality:
    ticker: str
    missing_fields: List[str]
    notes: List[str]


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pick_first(items: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in items:
            v = _to_float(items.get(k))
            if v is not None:
                return v
    return None


def _align_common_years(fin: Financials) -> List[str]:
    is_dates = set(fin.income_annual.items_by_date.keys())
    cf_dates = set(fin.cashflow_annual.items_by_date.keys())
    return sorted(is_dates.intersection(cf_dates))


def build_historical_drivers(fin: Financials, min_years: int = 5) -> Tuple[pd.DataFrame, DataQuality]:
    dates = _align_common_years(fin)
    notes: List[str] = []
    if len(dates) < min_years:
        notes.append(f"Only {len(dates)} overlapping annual years between IS and CF (min_years={min_years}).")

    rows: List[Dict[str, Any]] = []
    missing_required: List[str] = []

    IS_REV_KEYS = ["totalRevenue"]
    IS_EBIT_KEYS = ["ebit", "operatingIncome"]
    IS_TAX_KEYS = ["incomeTaxExpense", "taxProvision"]
    IS_PRETAX_KEYS = ["incomeBeforeTax"]
    IS_INTEREST_KEYS = ["interestExpense"]
    IS_DA_KEYS = ["depreciationAndAmortization"]
    CF_CFO_KEYS = ["totalCashFromOperatingActivities"]
    CF_DA_KEYS = ["depreciation"]
    CF_CAPEX_KEYS = ["capitalExpenditures"]
    CF_DNWC_KEYS = ["changeInWorkingCapital"]
    CF_FCF_KEYS = ["freeCashFlow"]

    for d in dates:
        is_items = fin.income_annual.items_by_date.get(d, {})
        cf_items = fin.cashflow_annual.items_by_date.get(d, {})

        revenue = _pick_first(is_items, IS_REV_KEYS)
        ebit = _pick_first(is_items, IS_EBIT_KEYS)
        pretax = _pick_first(is_items, IS_PRETAX_KEYS)
        tax = _pick_first(is_items, IS_TAX_KEYS)
        interest_expense = _pick_first(is_items, IS_INTEREST_KEYS)
        cfo = _pick_first(cf_items, CF_CFO_KEYS)

        da = _pick_first(cf_items, CF_DA_KEYS)
        if da is None:
            da = _pick_first(is_items, IS_DA_KEYS)
            if da is not None:
                notes.append(f"{fin.ticker} {d}: D&A from IS (CF depreciation missing).")

        capex_raw = _pick_first(cf_items, CF_CAPEX_KEYS)
        fcf_reported = _pick_first(cf_items, CF_FCF_KEYS)

        # ΔNWC: negate EODHD cash-flow convention; default to 0 if missing
        dnwc_cf_raw = _pick_first(cf_items, CF_DNWC_KEYS)
        if dnwc_cf_raw is not None:
            dnwc = -dnwc_cf_raw
        else:
            dnwc = 0.0  # Default: assume no incremental WC investment
            notes.append(f"{fin.ticker} {d}: ΔNWC missing, defaulted to 0.")

        capex_spend = abs(capex_raw) if capex_raw is not None else None

        tax_rate = None
        if tax is not None and pretax is not None and pretax > 0.0 and tax >= 0.0:
            tax_rate = tax / pretax
            tax_rate = max(0.0, min(0.6, tax_rate))

        nopat = None
        if ebit is not None:
            tr = tax_rate if tax_rate is not None else 0.21
            nopat = ebit * (1.0 - tr)
            if tax_rate is None:
                notes.append(f"{fin.ticker} {d}: tax_rate missing, used fallback 0.21 for NOPAT.")

        ufcf = None
        if nopat is not None and da is not None and capex_spend is not None:
            ufcf = nopat + da - capex_spend - dnwc

        fcf_gap = None
        if ufcf is not None and fcf_reported is not None:
            fcf_gap = ufcf - fcf_reported

        rows.append({
            "date": d, "currency": fin.currency, "revenue": revenue, "ebit": ebit,
            "pretax_income": pretax, "tax_expense": tax, "tax_rate": tax_rate,
            "interest_expense": interest_expense, "cfo": cfo, "da": da,
            "capex_spend": capex_spend, "dnwc": dnwc, "nopat": nopat, "ufcf": ufcf,
            "fcf_reported": fcf_reported, "ufcf_minus_reported_fcf": fcf_gap,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Only revenue, ebit, and da are truly required
    # dnwc defaults to 0 when missing; capex is needed for UFCF
    required_cols = ["revenue", "ebit", "da"]
    for col in required_cols:
        if col not in df.columns or df[col].isna().all():
            missing_required.append(col)

    # capex_spend is also needed but checked separately
    if "capex_spend" not in df.columns or df["capex_spend"].isna().all():
        missing_required.append("capex_spend")

    dq = DataQuality(ticker=fin.ticker, missing_fields=missing_required, notes=notes)
    return df, dq
