from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


MARGINAL_TAX_RATE_US = 0.21


@dataclass
class WACCInputs:
    risk_free_rate: float = 0.043
    equity_risk_premium: float = 0.045
    marginal_tax_rate: float = MARGINAL_TAX_RATE_US
    cost_of_debt: Optional[float] = None


SYNTHETIC_RATING_TABLE = [
    # Coverage breakpoints and spreads from Damodaran (January 2026)
    # Large non-financial service firms
    # Source: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html
    (-1e12,  0.20, "D",     0.1900),
    ( 0.20,  0.65, "C",     0.1600),
    ( 0.65,  0.80, "CC",    0.1261),
    ( 0.80,  1.25, "CCC",   0.0885),
    ( 1.25,  1.50, "B-",    0.0509),
    ( 1.50,  1.75, "B",     0.0321),
    ( 1.75,  2.00, "B+",    0.0275),
    ( 2.00,  2.25, "BB",    0.0184),
    ( 2.25,  2.50, "BB+",   0.0138),
    ( 2.50,  3.00, "BBB",   0.0111),
    ( 3.00,  4.25, "A-",    0.0089),
    ( 4.25,  5.50, "A",     0.0078),
    ( 5.50,  6.50, "A+",    0.0070),
    ( 6.50,  8.50, "AA",    0.0055),
    ( 8.50,  1e12, "AAA",   0.0040),
]


def _synthetic_spread(interest_coverage: float) -> tuple[str, float]:
    for min_cov, max_cov, rating, spread in SYNTHETIC_RATING_TABLE:
        if min_cov <= interest_coverage < max_cov:
            return rating, spread
    return "BBB", 0.0200


def compute_cost_of_equity(rf: float, beta: float, erp: float) -> float:
    return rf + beta * erp


def compute_wacc(
    market_cap: float, total_debt: float, beta: float,
    wacc_inputs: WACCInputs,
    effective_tax_rate: Optional[float] = None,
    interest_expense: Optional[float] = None,
    ebit: Optional[float] = None,
) -> Dict[str, Any]:
    rf = wacc_inputs.risk_free_rate
    erp = wacc_inputs.equity_risk_premium
    tax_for_wacc = wacc_inputs.marginal_tax_rate

    re = compute_cost_of_equity(rf, beta, erp)

    rd = wacc_inputs.cost_of_debt
    rd_method = "explicit"

    if rd is None:
        if (ebit is not None and interest_expense is not None
                and abs(float(interest_expense)) > 0):
            coverage = float(ebit) / abs(float(interest_expense))
            rating, spread = _synthetic_spread(coverage)
            rd = rf + spread
            rd_method = f"synthetic_rating ({rating}, coverage={coverage:.2f})"
        else:
            rd = rf + 0.02
            rd_method = "fallback (rf + 2%)"

    rd = max(rf, min(0.15, rd))

    d = float(total_debt)
    e = float(market_cap)
    v = d + e
    wd = d / v if v > 0 else 0.0
    we = e / v if v > 0 else 1.0

    wacc = we * re + wd * rd * (1.0 - tax_for_wacc)

    return {
        "wacc": wacc,
        "cost_of_equity": re,
        "cost_of_debt": rd,
        "cost_of_debt_method": rd_method,
        "weights": {"equity": we, "debt": wd},
        "inputs_used": {
            "rf": rf, "erp": erp, "beta": beta,
            "tax_for_wacc": tax_for_wacc,
            "tax_method": "marginal (TCJA 21%)",
        },
    }
