from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StatementSeries:
    """
    Stores statements as {period_end_date: lineitems_dict}
    Example: {"2024-09-28": {"totalRevenue": 383285000000, ...}, ...}
    """
    items_by_date: Dict[str, Dict[str, Any]]

    def dates_desc(self) -> List[str]:
        return sorted(self.items_by_date.keys(), reverse=True)

    def get(self, date: str, key: str, default: Any = None) -> Any:
        return self.items_by_date.get(date, {}).get(key, default)


@dataclass
class Financials:
    ticker: str
    currency: Optional[str]
    income_annual: StatementSeries
    cashflow_annual: StatementSeries
    balance_annual: Optional[StatementSeries] = None

    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None


def _safe_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _parse_statement_items(statement_node: Any) -> Dict[str, Dict[str, Any]]:
    """
    EODHD fundamentals typically nests statements like:
    Financials -> Income_Statement -> yearly -> {date: {lineitems...}}
    This function returns {date: {lineitems...}} or empty dict.
    """
    if not isinstance(statement_node, dict):
        return {}
    # Some APIs use 'yearly' or 'annual' keys; prefer 'yearly'
    yearly = statement_node.get("yearly") or statement_node.get("annual") or {}
    if not isinstance(yearly, dict):
        return {}
    # Ensure each value is a dict
    out: Dict[str, Dict[str, Any]] = {}
    for date, items in yearly.items():
        if isinstance(items, dict):
            out[str(date)] = items
    return out


def parse_eodhd_fundamentals(ticker: str, raw: Dict[str, Any]) -> Financials:
    # Currency can appear in General or Financials; we keep it optional
    currency = _safe_get(raw, ["General", "CurrencyCode"]) or _safe_get(raw, ["General", "Currency"])

    income_node = _safe_get(raw, ["Financials", "Income_Statement"])
    cashflow_node = _safe_get(raw, ["Financials", "Cash_Flow"])
    balance_node = _safe_get(raw, ["Financials", "Balance_Sheet"])

    income_annual = StatementSeries(_parse_statement_items(income_node))
    cashflow_annual = StatementSeries(_parse_statement_items(cashflow_node))
    balance_annual = StatementSeries(_parse_statement_items(balance_node)) if balance_node else None

    shares = _safe_get(raw, ["SharesStats", "SharesOutstanding"])
    market_cap = _safe_get(raw, ["Highlights", "MarketCapitalization"])
    beta = _safe_get(raw, ["Technicals", "Beta"])

    # Normalize numeric types if possible
    shares_outstanding = float(shares) if shares is not None else None
    market_cap_val = float(market_cap) if market_cap is not None else None
    beta_val = float(beta) if beta is not None else None

    return Financials(
        ticker=ticker,
        currency=currency,
        income_annual=income_annual,
        cashflow_annual=cashflow_annual,
        balance_annual=balance_annual,
        shares_outstanding=shares_outstanding,
        market_cap=market_cap_val,
        beta=beta_val,
    )
