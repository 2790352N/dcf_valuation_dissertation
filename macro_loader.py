"""
macro_loader.py — Load and match time-varying Rf and ERP to valuation dates.

The macro series CSV contains annual snapshots (as of January 1) of:
  - rf_10yr: US 10-Year Treasury yield (FRED DGS10 / Damodaran T.Bond Rate)
  - erp_damodaran: Damodaran implied ERP (FCFE method)

Matching rule: for a given as-of date, use the most recent January
observation that is <= the as-of date. This ensures we use information
that was available at the time of valuation, avoiding look-ahead bias.

Sources:
  Rf: FRED series DGS10 (https://fred.stlouisfed.org/series/DGS10)
  ERP: Damodaran (https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


class MacroSeriesLoader:
    def __init__(self, csv_path: str):
        """
        Load the macro series CSV.
        Expects columns: date, rf_10yr, erp_damodaran
        """
        self.path = Path(csv_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Macro series file not found: {self.path}")

        self.df = pd.read_csv(
            self.path,
            comment="#",
            parse_dates=["date"],
        )
        self.df = self.df.sort_values("date").reset_index(drop=True)

        # Validate required columns
        for col in ["date", "rf_10yr", "erp_damodaran"]:
            if col not in self.df.columns:
                raise ValueError(f"Macro series CSV missing required column: {col}")

    def get_rf_erp(self, as_of_date: str) -> Tuple[float, float, str]:
        """
        Look up the Rf and ERP for a given as-of date.

        Matching rule: most recent observation where date <= as_of_date.
        This avoids look-ahead bias (Hou, van Dijk and Zhang, 2012).

        Returns:
            (rf, erp, matched_date_str)

        Raises:
            ValueError if no matching date is found (as_of_date is before
            the earliest observation in the series).
        """
        target = pd.to_datetime(as_of_date)

        # Filter to observations <= as_of_date
        valid = self.df[self.df["date"] <= target]

        if valid.empty:
            raise ValueError(
                f"No macro series observation on or before {as_of_date}. "
                f"Earliest available: {self.df['date'].iloc[0].date()}"
            )

        # Take the most recent one
        row = valid.iloc[-1]
        rf = float(row["rf_10yr"])
        erp = float(row["erp_damodaran"])
        matched_date = str(row["date"].date())

        return rf, erp, matched_date

    def summary(self) -> str:
        """Print a summary of the loaded series for debugging."""
        lines = [f"Macro series: {len(self.df)} observations from {self.path.name}"]
        for _, row in self.df.iterrows():
            lines.append(
                f"  {row['date'].date()}: Rf={row['rf_10yr']:.4f}, ERP={row['erp_damodaran']:.4f}"
            )
        return "\n".join(lines)
