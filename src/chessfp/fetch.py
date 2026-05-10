"""chess.com Published-Data API client.

Polite by default: identifies itself with a contact email per chess.com's
guidelines, throttles to 1 req/sec, retries on 429/5xx with backoff,
and resumes by skipping monthly archives already on disk.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

API_BASE = "https://api.chess.com/pub"
USER_AGENT = "chessfp/0.0.1 (contact: theophilus.homawoo@mantiq.com)"

log = logging.getLogger(__name__)


@dataclass
class FetchStats:
    new_archives: int = 0
    skipped_archives: int = 0
    games: int = 0
    missing_handles: list[str] = field(default_factory=list)


class ChessComClient:
    def __init__(self, rate_limit_s: float = 1.0, user_agent: str = USER_AGENT, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.rate_limit_s = rate_limit_s
        self.timeout = timeout
        self._last_req = 0.0

    def _throttle(self) -> None:
        wait = self.rate_limit_s - (time.monotonic() - self._last_req)
        if wait > 0:
            time.sleep(wait)
        self._last_req = time.monotonic()

    def _get(self, url: str) -> requests.Response:
        for attempt in range(5):
            self._throttle()
            r = self.session.get(url, timeout=self.timeout)
            if r.status_code == 429:
                backoff = min(60, 5 * 2**attempt)
                log.warning("429 from %s — sleeping %ss", url, backoff)
                time.sleep(backoff)
                continue
            if r.status_code in (500, 502, 503, 504):
                backoff = min(30, 2**attempt)
                log.warning("%s from %s — retrying in %ss", r.status_code, url, backoff)
                time.sleep(backoff)
                continue
            return r
        return r  # last response, even if it was retryable

    def list_archives(self, handle: str) -> list[str]:
        r = self._get(f"{API_BASE}/player/{handle}/games/archives")
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get("archives", [])

    def fetch_archive(self, archive_url: str) -> dict:
        r = self._get(archive_url)
        r.raise_for_status()
        return r.json()


def archive_url_to_yyyymm(archive_url: str) -> str:
    parts = archive_url.rstrip("/").split("/")
    return f"{parts[-2]}-{parts[-1]}"


def fetch_player(
    client: ChessComClient,
    player: dict,
    out_dir: Path,
    since: str | None = None,
) -> FetchStats:
    """Fetch all monthly archives for a player and any known aliases.

    Layout: out_dir/{player_id}/{handle}/{YYYY-MM}.json

    Args:
        since: optional "YYYY-MM" cutoff; only archives >= this month are fetched.
    """
    stats = FetchStats()
    handles = [player["handle"], *player.get("aliases", [])]
    player_dir = out_dir / player["id"]
    player_dir.mkdir(parents=True, exist_ok=True)

    for handle in handles:
        archives = client.list_archives(handle)
        if not archives:
            stats.missing_handles.append(handle)
            log.info("  no archives for handle %r (skip)", handle)
            continue
        handle_dir = player_dir / handle
        handle_dir.mkdir(exist_ok=True)
        for archive_url in archives:
            yyyymm = archive_url_to_yyyymm(archive_url)
            if since is not None and yyyymm < since:
                continue
            out_path = handle_dir / f"{yyyymm}.json"
            if out_path.exists() and out_path.stat().st_size > 0:
                stats.skipped_archives += 1
                continue
            try:
                data = client.fetch_archive(archive_url)
            except requests.HTTPError as e:
                log.warning("  failed %s: %s", archive_url, e)
                continue
            out_path.write_text(json.dumps(data))
            stats.new_archives += 1
            stats.games += len(data.get("games", []))
    return stats
