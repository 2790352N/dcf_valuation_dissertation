"""
historical_mcap.py — Fetch historical market capitalisation from EODHD.

Uses the /api/historical-market-cap/ endpoint (premium plan).
Returns weekly market cap data; we pick the nearest observation
to the requested as-of date.

Note: EODHD returns this data as a dict with string-integer keys
(e.g., {"0": {"date": ..., "value": ...}, "1": {...}, ...})
rather than a JSON array. We handle both formats.

Source: https://eodhd.com/financial-apis/historical-market-capitalization-api
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_response(data: Any) -> List[Dict[str, Any]]:
    """
    EODHD historical-market-cap returns either:
      - a list of dicts (documented format)
      - a dict with string-integer keys {"0": {...}, "1": {...}, ...}
    Normalize to a list in either case.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Check if it's a string-keyed indexed dict
        # vs an error response like {"error": "..."}
        if "error" in data or "message" in data:
            return []
        # Convert dict values to list, sorted by key
        try:
            sorted_keys = sorted(data.keys(), key=lambda k: int(k))
            return [data[k] for k in sorted_keys if isinstance(data[k], dict)]
        except (ValueError, TypeError):
            # Keys aren't numeric — might be an unexpected format
            # Try just returning all dict values that look like observations
            return [v for v in data.values() if isinstance(v, dict) and "date" in v]
    return []


class HistoricalMcapClient:
    def __init__(self, api_key: str, cache_dir: str = "data/raw", timeout_sec: int = 30):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec

    def _cache_path(self, ticker: str, date_from: str, date_to: str) -> Path:
        safe = ticker.replace("/", "_").replace(".", "_")
        return self.cache_dir / f"{safe}_hist_mcap_{date_from}_{date_to}.json"

    def fetch_historical_mcap(
        self,
        ticker: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical market cap series from EODHD.
        Returns list of dicts like: {"date": "2024-01-29", "value": 3005193000000}
        """
        # Use date params in cache key to avoid stale broad caches
        cf = date_from or "none"
        ct = date_to or "none"
        cache_path = self._cache_path(ticker, cf, ct)

        if use_cache and not force_refresh and cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return _normalize_response(raw)

        url = f"https://eodhd.com/api/historical-market-cap/{ticker}"
        params: Dict[str, str] = {
            "api_token": self.api_key,
            "fmt": "json",
        }
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to

        try:
            r = requests.get(url, params=params, timeout=self.timeout_sec)
            if r.status_code != 200:
                print(f"  [WARN] Historical mcap API returned {r.status_code} for {ticker}")
                return []

            raw_data = r.json()

            # Cache the raw response
            cache_path.write_text(json.dumps(raw_data), encoding="utf-8")

            return _normalize_response(raw_data)

        except Exception as e:
            print(f"  [WARN] Historical mcap fetch failed for {ticker}: {e}")
            return []

    def get_mcap_near_date(
        self,
        ticker: str,
        target_date: str,
        window_days: int = 30,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Find the nearest historical market cap observation within window_days
        of target_date.

        Returns (date_used, market_cap) or (None, None).
        """
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        date_from = (dt - timedelta(days=window_days)).isoformat()
        date_to = (dt + timedelta(days=window_days)).isoformat()

        data = self.fetch_historical_mcap(
            ticker=ticker,
            date_from=date_from,
            date_to=date_to,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

        if not data:
            return None, None

        best_date = None
        best_mcap = None
        best_dist = None

        for obs in data:
            d = obs.get("date")
            mcap = _safe_float(obs.get("value"))
            if d is None or mcap is None:
                continue

            try:
                obs_date = datetime.strptime(d, "%Y-%m-%d").date()
            except ValueError:
                continue

            dist = abs((obs_date - dt).days)
            if best_dist is None or dist < best_dist:
                best_date = d
                best_mcap = mcap
                best_dist = dist

        return best_date, best_mcap
