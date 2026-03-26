from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np


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


def extract_cash_and_debt(balance_items: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Returns cash, total_debt, net_debt (debt - cash), if possible.
    Uses common EODHD/Yahoo-style field names; adjust keys once you confirm exact ones in your JSON.
    """
    CASH_KEYS = [
        "cash",
        "cashAndEquivalents",
        "cashAndCashEquivalents",
        "cashAndCashEquivalentsAtCarryingValue",
        "cashCashEquivalentsAndShortTermInvestments",        
    ]

    SHORT_DEBT_KEYS = [
        "shortTermDebt",
        "shortLongTermDebt",
    ]

    LONG_DEBT_KEYS = [
        "longTermDebt",
        "longTermDebtTotal",
        "capitalLeaseObligations",
    ]

    TOTAL_DEBT_KEYS = [
        "shortLongTermDebtTotal",
        "totalDebt",
    ]

    NET_DEBT_KEYS = [
        "netDebt",
    ]


    cash = _pick_first(balance_items, CASH_KEYS)

    # Prefer provided netDebt if present (some providers already compute it)
    net_debt = _pick_first(balance_items, NET_DEBT_KEYS)

    total_debt = _pick_first(balance_items, TOTAL_DEBT_KEYS)
    if total_debt is None:
        short_debt = _pick_first(balance_items, SHORT_DEBT_KEYS) or 0.0
        long_debt = _pick_first(balance_items, LONG_DEBT_KEYS) or 0.0
        if short_debt or long_debt:
            total_debt = float(short_debt) + float(long_debt)

    # If netDebt wasn't provided, compute it
    if net_debt is None and total_debt is not None and cash is not None:
        net_debt = float(total_debt) - float(cash)

    return {"cash": cash, "total_debt": total_debt, "net_debt": net_debt}



