from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class EODHDConfig:
    api_key: str
    base_url: str = "https://eodhd.com/api/fundamentals"
    timeout_sec: int = 30
    max_retries: int = 4
    backoff_sec: float = 1.5  # exponential backoff multiplier


class EODHDFundamentalsClient:
    def __init__(self, config: EODHDConfig, cache_dir: str = "data/raw") -> None:
        self.config = config
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "dcf_project/1.0"})

    def _cache_file(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_")
        return self.cache_path / f"{safe}_fundamentals.json"

    def get_fundamentals(
        self,
        ticker: str,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch fundamentals JSON for a ticker. Uses disk cache by default.
        """
        cache_file = self._cache_file(ticker)

        if use_cache and not force_refresh and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        url = f"{self.config.base_url}/{ticker}"
        params = {"api_token": self.config.api_key, "fmt": "json"}

        last_err: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                print("DEBUG fundamentals api key prefix:", str(self.config.api_key)[:6], "len=", len(str(self.config.api_key)))
                print("DEBUG fundamentals url:", url)
                print("DEBUG fundamentals params:", params)
                resp = self.session.get(url, params=params, timeout=self.config.timeout_sec)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                cache_file.write_text(json.dumps(data), encoding="utf-8")
                return data
            except Exception as e:
                last_err = e
                sleep = (self.config.backoff_sec ** (attempt - 1))
                time.sleep(sleep)

        raise RuntimeError(f"Failed to fetch fundamentals for {ticker}. Last error: {last_err}")
