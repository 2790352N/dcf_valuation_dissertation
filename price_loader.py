from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class EODHDPriceConfig:
    api_key: str = "api_key"
    base_url: str = "https://eodhd.com/api/eod"
    cache_dir: str = "data/raw"
    timeout_sec: int = 30


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


class EODHDPriceClient:
    def __init__(self, cfg: EODHDPriceConfig):
        self.cfg = cfg
        Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, date_from: str, date_to: str) -> Path:
        safe = ticker.replace("/", "_")
        return Path(self.cfg.cache_dir) / f"{safe}_eod_{date_from}_{date_to}.json"

    def get_eod_range(
        self,
        ticker: str,
        date_from: str,
        date_to: str,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> list[Dict[str, Any]]:
        """
        Fetch EOD daily candles in [date_from, date_to].
        Returns list of dicts like: {"date":"YYYY-MM-DD","close":...,...}
        """
        cache_path = self._cache_path(ticker, date_from, date_to)

        if use_cache and (not force_refresh) and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{self.cfg.base_url}/{ticker}"
        params = {
            "api_token": self.cfg.api_key,
            "fmt": "json",
            "from": date_from,
            "to": date_to,
        }
        print("DEBUG price api key prefix:", str(self.cfg.api_key)[:6], "len=", len(str(self.cfg.api_key)))
        r = requests.get(url, params=params, timeout=self.cfg.timeout_sec)
        if r.status_code != 200:
            return []

        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected EOD response type: {type(data)}")

        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def get_close_near_date(
        self,
        ticker: str,
        target_date: str,
        window_days: int = 7,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Find the nearest available trading day close within +/- window_days around target_date.
        Returns (date_used, close) or (None, None).
        """
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        date_from = (dt - timedelta(days=window_days)).isoformat()
        date_to = (dt + timedelta(days=window_days)).isoformat()

        candles = self.get_eod_range(
            ticker=ticker,
            date_from=date_from,
            date_to=date_to,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

        if not candles:
            return None, None

        # pick nearest by absolute day difference
        best = None
        best_dist = None
        for c in candles:
            d = c.get("date")
            close = _safe_float(c.get("close"))
            if not d or close is None:
                continue
            cd = datetime.strptime(d, "%Y-%m-%d").date()
            dist = abs((cd - dt).days)
            if best is None or dist < best_dist:
                best = (d, close)
                best_dist = dist

        if best is None:
            return None, None
        return best

