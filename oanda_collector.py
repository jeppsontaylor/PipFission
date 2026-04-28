#!/usr/bin/env python3
"""
oanda_collector.py

High-quality OANDA market data collector:
- Streams top-of-book pricing via:
    GET /v3/accounts/{accountID}/pricing/stream   (streaming host)
  Notes:
    - OANDA provides at most ~4 prices/sec per instrument (~250ms).
    - Heartbeats are sent every ~5 seconds.
- Polls FXLabs sentiment endpoints via REST host:
    GET /labs/v1/orderbook_data
    GET /labs/v1/historical_position_ratios

Outputs:
- JSON Lines (one JSON object per line), rotated hourly:
    pricing_raw-YYYYMMDD_HH.jsonl[.gz]
    pricing_1s-YYYYMMDD_HH.jsonl[.gz]
    labs_orderbook-YYYYMMDD_HH.jsonl[.gz]
    labs_position_ratios-YYYYMMDD_HH.jsonl[.gz]
- run_config.json in the run directory

Requires:
  pip install aiohttp python-dotenv

Usage:
  # The .env file will be loaded automatically if present
  python oanda_collector.py --instruments EUR_USD,USD_JPY --env practice

Optional:
  export OANDA_ACCOUNT_ID="..."   # if not set, we auto-discover via /v3/accounts
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import gzip
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import aiohttp

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# -----------------------------
# JSON: prefer orjson if present
# -----------------------------
try:
    import orjson  # type: ignore

    def json_dumps_bytes(obj: Any) -> bytes:
        return orjson.dumps(obj)

    def json_loads(data: bytes) -> Any:
        return orjson.loads(data)

except Exception:
    def json_dumps_bytes(obj: Any) -> bytes:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def json_loads(data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


# -----------------------------
# OANDA environments
# -----------------------------
@dataclasses.dataclass(frozen=True)
class OandaEnvironment:
    name: str
    rest_url: str
    stream_url: str


ENVIRONMENTS: Dict[str, OandaEnvironment] = {
    "practice": OandaEnvironment(
        name="practice",
        rest_url="https://api-fxpractice.oanda.com",
        stream_url="https://stream-fxpractice.oanda.com",
    ),
    "live": OandaEnvironment(
        name="live",
        rest_url="https://api-fxtrade.oanda.com",
        stream_url="https://stream-fxtrade.oanda.com",
    ),
}


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_instrument(s: str) -> str:
    """
    Accepts:
      EUR_USD, EUR/USD, eurusd, eur-usd, etc.
    Returns:
      EUR_USD
    """
    s = s.strip().upper()
    if not s:
        raise ValueError("Empty instrument symbol")

    # Common forms: "EUR/USD"
    s = s.replace("/", "_").replace("-", "_").replace(" ", "")
    # If "EURUSD" -> "EUR_USD"
    if "_" not in s and len(s) == 6 and s.isalpha():
        s = f"{s[:3]}_{s[3:]}"
    return s


def parse_instruments(arg: str) -> List[str]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("No instruments provided")
    return [normalize_instrument(p) for p in parts]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class ApiError(RuntimeError):
    pass


# -----------------------------
# Rotating JSONL writer
# -----------------------------
class RotatingJsonlWriter:
    """
    Writes JSONL (optionally gzipped) and rotates files hourly:
      {kind}-YYYYMMDD_HH.jsonl[.gz]
    """

    def __init__(self, out_dir: Path, kind: str, gzip_enabled: bool, flush_every: int = 50):
        self.out_dir = out_dir
        self.kind = kind
        self.gzip_enabled = gzip_enabled
        self.flush_every = max(1, flush_every)

        self._current_hour: Optional[str] = None
        self._fh = None
        self._count_since_flush = 0

    def _hour_key(self, t: dt.datetime) -> str:
        return t.strftime("%Y%m%d_%H")

    def _path_for(self, hour_key: str) -> Path:
        suffix = ".jsonl.gz" if self.gzip_enabled else ".jsonl"
        return self.out_dir / f"{self.kind}-{hour_key}{suffix}"

    def _open_for_hour(self, hour_key: str) -> None:
        # Close old
        self.close()

        path = self._path_for(hour_key)
        ensure_dir(self.out_dir)

        if self.gzip_enabled:
            # compresslevel=1 = fast; file will contain concatenated gzip members if appended
            self._fh = gzip.open(path, mode="ab", compresslevel=1)
        else:
            self._fh = open(path, mode="ab")

        self._current_hour = hour_key
        self._count_since_flush = 0

    def write_record(self, record: Dict[str, Any]) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        hour_key = self._hour_key(now)
        if self._current_hour != hour_key or self._fh is None:
            self._open_for_hour(hour_key)

        line = json_dumps_bytes(record) + b"\n"
        self._fh.write(line)
        self._count_since_flush += 1

        if self._count_since_flush >= self.flush_every:
            self._fh.flush()
            self._count_since_flush = 0

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None


# -----------------------------
# HTTP helpers with retries
# -----------------------------
async def fetch_json_with_retries(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, str]] = None,
    timeout_s: float = 10.0,
    retries: int = 6,
) -> Any:
    """
    Robust JSON fetch:
      - Retries on 429 and common 5xx
      - Exponential backoff + jitter
    """
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    for attempt in range(retries):
        try:
            async with session.request(method, url, headers=headers, params=params, timeout=timeout) as resp:
                body = await resp.read()

                if resp.status == 200:
                    if not body:
                        return None
                    return json_loads(body)

                # Retryable?
                if resp.status in (429, 500, 502, 503, 504):
                    backoff = min(30.0, (2 ** attempt) * 0.5) + random.random() * 0.25
                    logging.warning("Retryable HTTP %s from %s (attempt %d/%d). Sleeping %.2fs",
                                    resp.status, url, attempt + 1, retries, backoff)
                    await asyncio.sleep(backoff)
                    continue

                # Non-retryable
                msg = body.decode("utf-8", errors="replace")[:5000]
                raise ApiError(f"HTTP {resp.status} for {url}: {msg}")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            backoff = min(30.0, (2 ** attempt) * 0.5) + random.random() * 0.25
            logging.warning("Network error %r on %s (attempt %d/%d). Sleeping %.2fs",
                            e, url, attempt + 1, retries, backoff)
            await asyncio.sleep(backoff)

    raise ApiError(f"Failed after {retries} attempts: {url}")


# -----------------------------
# OANDA account discovery
# -----------------------------
async def discover_first_account_id(
    session: aiohttp.ClientSession,
    env: OandaEnvironment,
    headers: Dict[str, str],
) -> str:
    """
    Calls GET /v3/accounts and returns the first account id.
    """
    url = f"{env.rest_url}/v3/accounts"
    data = await fetch_json_with_retries(session, "GET", url, headers=headers, timeout_s=10.0)
    if not isinstance(data, dict) or "accounts" not in data:
        raise ApiError(f"Unexpected /v3/accounts response: {data}")

    accounts = data.get("accounts", [])
    if not accounts:
        raise ApiError("No accounts returned by /v3/accounts")

    first = accounts[0]
    account_id = first.get("id")
    if not account_id:
        raise ApiError(f"Unexpected account object: {first}")

    return str(account_id)


# -----------------------------
# Streaming pricing
# -----------------------------
async def pricing_stream_task(
    session: aiohttp.ClientSession,
    env: OandaEnvironment,
    account_id: str,
    instruments: List[str],
    headers: Dict[str, str],
    out_queue: asyncio.Queue,
    last_price: Dict[str, Dict[str, Any]],
    stop_event: asyncio.Event,
    snapshot: bool = True,
    include_home_conversions: bool = False,
) -> None:
    """
    Maintains a persistent connection to the OANDA pricing stream and pushes records into out_queue.
    Auto-reconnects on disconnect.
    """
    instruments_csv = ",".join(instruments)
    url = f"{env.stream_url}/v3/accounts/{account_id}/pricing/stream"
    params = {
        "instruments": instruments_csv,
        "snapshot": "true" if snapshot else "false",
        "includeHomeConversions": "true" if include_home_conversions else "false",
    }

    # Important: no total timeout for streaming connections
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)

    while not stop_event.is_set():
        try:
            logging.info("Connecting pricing stream: %s instruments=%s", url, instruments_csv)
            async with session.get(url, headers=headers, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    body = (await resp.read()).decode("utf-8", errors="replace")[:5000]
                    raise ApiError(f"Pricing stream HTTP {resp.status}: {body}")

                buffer = b""
                async for chunk in resp.content.iter_chunked(4096):
                    if stop_event.is_set():
                        break
                    if not chunk:
                        continue

                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            msg = json_loads(line)
                        except Exception:
                            # Log parse errors but keep running
                            await out_queue.put({
                                "kind": "error",
                                "ts": utc_now_iso(),
                                "source": "pricing_stream",
                                "error": "json_parse_error",
                                "raw": line[:500].decode("utf-8", errors="replace"),
                            })
                            continue

                        # Heartbeats have type=HEARTBEAT (documented)
                        mtype = msg.get("type")
                        if mtype == "HEARTBEAT":
                            await out_queue.put({
                                "kind": "pricing_raw",
                                "ts": utc_now_iso(),
                                "stream": "heartbeat",
                                "data": msg,
                            })
                            continue

                        # Price messages: keep raw + update last_price for 1s snapshots
                        inst = msg.get("instrument")
                        if isinstance(inst, str):
                            last_price[inst] = msg

                        await out_queue.put({
                            "kind": "pricing_raw",
                            "ts": utc_now_iso(),
                            "stream": "price",
                            "data": msg,
                        })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.warning("Pricing stream error: %r (reconnecting soon)", e)
            await out_queue.put({
                "kind": "error",
                "ts": utc_now_iso(),
                "source": "pricing_stream",
                "error": repr(e),
            })
            # Small backoff before reconnect
            await asyncio.sleep(1.0)


async def pricing_1s_snapshot_task(
    instruments: List[str],
    last_price: Dict[str, Dict[str, Any]],
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    """
    Once per second, emits a compact top-of-book snapshot per instrument
    based on the most recent streamed message we've seen.
    """
    def best_px(side: Any) -> Optional[str]:
        # side should be a list like [{"liquidity":..., "price":"..."}]
        if isinstance(side, list) and side:
            p = side[0].get("price")
            return str(p) if p is not None else None
        return None

    while not stop_event.is_set():
        # Align to next second boundary
        now = time.time()
        next_sec = int(now) + 1
        await asyncio.sleep(max(0.0, next_sec - now))

        ts = utc_now_iso()
        # Copy references quickly (avoid long iteration holding shared dict)
        for inst in instruments:
            msg = last_price.get(inst)
            if not msg:
                continue

            bid = best_px(msg.get("bids"))
            ask = best_px(msg.get("asks"))
            rec = {
                "kind": "pricing_1s",
                "ts": ts,
                "instrument": inst,
                "source_time": msg.get("time"),
                "bid": bid,
                "ask": ask,
                "closeoutBid": msg.get("closeoutBid"),
                "closeoutAsk": msg.get("closeoutAsk"),
                "status": msg.get("status"),
            }
            await out_queue.put(rec)


# -----------------------------
# FXLabs polling
# -----------------------------
async def labs_orderbook_poll_task(
    session: aiohttp.ClientSession,
    env: OandaEnvironment,
    instruments: List[str],
    headers: Dict[str, str],
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    poll_interval_s: float = 60.0,
    period_s: int = 3600,
    log_full_response: bool = False,
) -> None:
    """
    Polls /labs/v1/orderbook_data and logs only NEW snapshots (deduped),
    unless log_full_response=True.
    """
    url = f"{env.rest_url}/labs/v1/orderbook_data"
    last_logged_ts: Dict[str, int] = {}

    while not stop_event.is_set():
        start = time.time()
        for inst in instruments:
            if stop_event.is_set():
                break
            params = {"instrument": inst, "period": str(period_s)}
            try:
                data = await fetch_json_with_retries(session, "GET", url, headers=headers, params=params, timeout_s=15.0)
                if not isinstance(data, dict) or not data:
                    continue

                if log_full_response:
                    await out_queue.put({
                        "kind": "labs_orderbook",
                        "ts": utc_now_iso(),
                        "instrument": inst,
                        "period": period_s,
                        "data": data,
                        "mode": "full",
                    })
                    continue

                # Keys are snapshot timestamps (strings). Take latest.
                snap_ts = max(int(k) for k in data.keys() if str(k).isdigit())
                if last_logged_ts.get(inst) == snap_ts:
                    continue

                last_logged_ts[inst] = snap_ts
                snap = data.get(str(snap_ts), {})
                await out_queue.put({
                    "kind": "labs_orderbook",
                    "ts": utc_now_iso(),
                    "instrument": inst,
                    "period": period_s,
                    "snapshot_unix": snap_ts,
                    "data": snap,
                    "mode": "latest_only",
                })

            except Exception as e:
                await out_queue.put({
                    "kind": "error",
                    "ts": utc_now_iso(),
                    "source": "labs_orderbook_data",
                    "instrument": inst,
                    "error": repr(e),
                })

        elapsed = time.time() - start
        sleep_for = max(0.0, poll_interval_s - elapsed)
        await asyncio.sleep(sleep_for)


async def labs_position_ratios_poll_task(
    session: aiohttp.ClientSession,
    env: OandaEnvironment,
    instruments: List[str],
    headers: Dict[str, str],
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    poll_interval_s: float = 60.0,
    period_s: int = 86400,
    log_full_response: bool = False,
) -> None:
    """
    Polls /labs/v1/historical_position_ratios, logs only NEW latest points (deduped),
    unless log_full_response=True.

    Response example (simplified):
      {"data":{"EUR_USD":{"data":[[timestamp,long_ratio,rate],...], "label":"EUR/USD"}}}
    """
    url = f"{env.rest_url}/labs/v1/historical_position_ratios"
    last_logged_ts: Dict[str, int] = {}

    while not stop_event.is_set():
        start = time.time()
        for inst in instruments:
            if stop_event.is_set():
                break
            params = {"instrument": inst, "period": str(period_s)}
            try:
                data = await fetch_json_with_retries(session, "GET", url, headers=headers, params=params, timeout_s=15.0)
                if not isinstance(data, dict) or not data:
                    continue

                if log_full_response:
                    await out_queue.put({
                        "kind": "labs_position_ratios",
                        "ts": utc_now_iso(),
                        "instrument": inst,
                        "period": period_s,
                        "data": data,
                        "mode": "full",
                    })
                    continue

                # Navigate to latest datapoint
                payload = data.get("data", {}).get(inst, {})
                series = payload.get("data", [])
                if not isinstance(series, list) or not series:
                    continue

                latest = series[-1]
                if not (isinstance(latest, list) and len(latest) >= 3):
                    continue

                snap_ts = int(latest[0])
                if last_logged_ts.get(inst) == snap_ts:
                    continue

                last_logged_ts[inst] = snap_ts
                await out_queue.put({
                    "kind": "labs_position_ratios",
                    "ts": utc_now_iso(),
                    "instrument": inst,
                    "period": period_s,
                    "snapshot_unix": snap_ts,
                    "long_position_ratio": latest[1],
                    "rate": latest[2],
                    "label": payload.get("label"),
                    "mode": "latest_only",
                })

            except Exception as e:
                await out_queue.put({
                    "kind": "error",
                    "ts": utc_now_iso(),
                    "source": "labs_historical_position_ratios",
                    "instrument": inst,
                    "error": repr(e),
                })

        elapsed = time.time() - start
        sleep_for = max(0.0, poll_interval_s - elapsed)
        await asyncio.sleep(sleep_for)



# -----------------------------
# V20 Polling (OrderBook / PositionBook)
# -----------------------------
async def v20_orderbook_poll_task(
    session: aiohttp.ClientSession,
    env: OandaEnvironment,
    instruments: List[str],
    headers: Dict[str, str],
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    poll_interval_s: float = 60.0,
    log_full_response: bool = True, # Default to full because it's the "richest" data
) -> None:
    """
    Polls /v3/instruments/{instrument}/orderBook.
    """
    last_time: Dict[str, str] = {} # dedupe by "time" field in response

    while not stop_event.is_set():
        start = time.time()
        for inst in instruments:
            if stop_event.is_set():
                break
            url = f"{env.rest_url}/v3/instruments/{inst}/orderBook"
            try:
                data = await fetch_json_with_retries(session, "GET", url, headers=headers, timeout_s=15.0)
                if not isinstance(data, dict):
                    continue
                
                # Check for "orderBook" key
                bk = data.get("orderBook")
                if not bk:
                    # Might be 404 or empty if not supported
                    continue

                t = bk.get("time")
                if t and last_time.get(inst) == t:
                    continue
                if t:
                    last_time[inst] = t

                await out_queue.put({
                    "kind": "v20_orderbook",
                    "ts": utc_now_iso(),
                    "instrument": inst,
                    "data": bk,
                })

            except Exception as e:
                 # Often 404/400 if instrument doesn't support it
                await out_queue.put({
                    "kind": "error",
                    "ts": utc_now_iso(),
                    "source": "v20_orderbook",
                    "instrument": inst,
                    "error": repr(e),
                })

        elapsed = time.time() - start
        sleep_for = max(0.0, poll_interval_s - elapsed)
        await asyncio.sleep(sleep_for)


async def v20_positionbook_poll_task(
    session: aiohttp.ClientSession,
    env: OandaEnvironment,
    instruments: List[str],
    headers: Dict[str, str],
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    poll_interval_s: float = 60.0,
    log_full_response: bool = True,
) -> None:
    """
    Polls /v3/instruments/{instrument}/positionBook.
    """
    last_time: Dict[str, str] = {}
    logging.info(f"Starting position book polling for: {instruments}")

    while not stop_event.is_set():
        start = time.time()
        for inst in instruments:
            if stop_event.is_set():
                break
            url = f"{env.rest_url}/v3/instruments/{inst}/positionBook"
            try:
                # logging.info(f"Polling PB for {inst}...")
                data = await fetch_json_with_retries(session, "GET", url, headers=headers, timeout_s=15.0)
                if not isinstance(data, dict):
                    logging.warning(f"PB data for {inst} is not a dict: {type(data)}")
                    continue

                bk = data.get("positionBook")
                if not bk:
                    logging.warning(f"PB data for {inst} missing 'positionBook' key. Keys: {list(data.keys())}")
                    continue

                t = bk.get("time")
                if t and last_time.get(inst) == t:
                    # logging.info(f"PB data for {inst} unchanged (time={t}). Skipping.")
                    continue
                if t:
                    last_time[inst] = t
                
                logging.info(f"Got NEW PB data for {inst} (time={t})")

                await out_queue.put({
                    "kind": "v20_positionbook",
                    "ts": utc_now_iso(),
                    "instrument": inst,
                    "data": bk,
                })

            except Exception as e:
                logging.error(f"Error polling PB for {inst}: {e}")
                await out_queue.put({
                    "kind": "error",
                    "ts": utc_now_iso(),
                    "source": "v20_positionbook",
                    "instrument": inst,
                    "error": repr(e),
                })

        elapsed = time.time() - start
        sleep_for = max(0.0, poll_interval_s - elapsed)
        await asyncio.sleep(sleep_for)


# -----------------------------
# Writer loop
# -----------------------------
async def writer_task(
    run_dir: Path,
    gzip_enabled: bool,
    flush_every: int,
    queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    writers = {
        "pricing_raw": RotatingJsonlWriter(run_dir, "pricing_raw", gzip_enabled, flush_every),
        "pricing_1s": RotatingJsonlWriter(run_dir, "pricing_1s", gzip_enabled, flush_every),
        "labs_orderbook": RotatingJsonlWriter(run_dir, "labs_orderbook", gzip_enabled, flush_every),
        "labs_position_ratios": RotatingJsonlWriter(run_dir, "labs_position_ratios", gzip_enabled, flush_every),
        "v20_orderbook": RotatingJsonlWriter(run_dir, "v20_orderbook", gzip_enabled, flush_every),
        "v20_positionbook": RotatingJsonlWriter(run_dir, "v20_positionbook", gzip_enabled, flush_every),
        "error": RotatingJsonlWriter(run_dir, "errors", gzip_enabled, flush_every),
    }

    try:
        while not stop_event.is_set() or not queue.empty():
            try:
                rec = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            kind = rec.get("kind", "error")
            w = writers.get(kind, writers["error"])
            w.write_record(rec)
            queue.task_done()

    finally:
        for w in writers.values():
            w.close()


# -----------------------------
# Main
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OANDA market data + sentiment logger (pricing stream + FXLabs + V20).")
    p.add_argument("--instruments", required=True, help="Comma-separated instruments (EUR_USD,USD_JPY or EURUSD).")
    p.add_argument("--env", choices=["practice", "live"], default="practice", help="OANDA environment.")
    p.add_argument("--account-id", default=os.environ.get("OANDA_ACCOUNT_ID", ""),
                   help="OANDA account ID. If omitted, auto-discovers the first account via /v3/accounts.")
    p.add_argument("--token", default=os.environ.get("OANDA_API_TOKEN", ""),
                   help="OANDA API token. If omitted, uses env var OANDA_API_TOKEN.")

    p.add_argument("--out", default="oanda_data", help="Base output directory.")
    p.add_argument("--gzip", action="store_true", help="Gzip-compress JSONL logs.")
    p.add_argument("--flush-every", type=int, default=50, help="Flush file buffers every N records.")

    p.add_argument("--disable-1s-snapshot", action="store_true", help="Disable derived 1-second pricing snapshots.")
    p.add_argument("--no-stream-snapshot", action="store_true",
                   help="Disable initial pricing snapshot on stream connect (snapshot=false).")

    # FXLabs polling knobs
    p.add_argument("--labs-orderbook", action="store_true", help="Enable FXLabs orderbook_data polling.")
    p.add_argument("--labs-orderbook-poll", type=float, default=60.0, help="Orderbook poll interval (seconds).")
    p.add_argument("--labs-orderbook-period", type=int, default=3600, help="Orderbook 'period' query param (seconds).")

    p.add_argument("--labs-position-ratios", action="store_true", help="Enable FXLabs historical_position_ratios polling.")
    p.add_argument("--labs-position-poll", type=float, default=60.0, help="Position ratios poll interval (seconds).")
    p.add_argument("--labs-position-period", type=int, default=86400, help="Position ratios 'period' query param (seconds).")

    p.add_argument("--labs-log-full", action="store_true",
                   help="Log full FXLabs responses each poll (bigger files). Default logs only newest snapshot.")
    
    # V20 Polling knobs
    p.add_argument("--v20-books", action="store_true", help="Enable V20 OrderBook and PositionBook polling (Richest Data).")
    p.add_argument("--v20-poll", type=float, default=60.0, help="V20 Book poll interval (seconds).")

    return p


async def amain(args: argparse.Namespace) -> int:
    token = args.token.strip()
    if not token:
        print("ERROR: Missing API token. Set OANDA_API_TOKEN or pass --token.", file=sys.stderr)
        return 2

    env = ENVIRONMENTS[args.env]
    instruments = parse_instruments(args.instruments)

    # Run directory
    run_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / run_stamp
    ensure_dir(run_dir)

    # Logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    stop_event = asyncio.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=20000)
    last_price: Dict[str, Dict[str, Any]] = {}

    # Write config for reproducibility
    (run_dir / "run_config.json").write_bytes(
        json_dumps_bytes({
            "created_at": utc_now_iso(),
            "env": env.name,
            "rest_url": env.rest_url,
            "stream_url": env.stream_url,
            "instruments": instruments,
            "gzip": bool(args.gzip),
            "labs_orderbook": bool(args.labs_orderbook),
            "labs_position_ratios": bool(args.labs_position_ratios),
            "labs_log_full": bool(args.labs_log_full),
            "v20_books": bool(args.v20_books),
        })
    )

    # Graceful shutdown signals
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        if not stop_event.is_set():
            logging.info("Shutdown requested...")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Windows fallback: signal handlers limited
            pass

    async with aiohttp.ClientSession() as session:
        account_id = args.account_id.strip()
        if not account_id:
            logging.info("Discovering account id via /v3/accounts ...")
            account_id = await discover_first_account_id(session, env, headers)
            logging.info("Using account id: %s", account_id)

        # Start writer
        writer = asyncio.create_task(writer_task(run_dir, args.gzip, args.flush_every, queue, stop_event))

        # Start pricing stream
        stream = asyncio.create_task(
            pricing_stream_task(
                session=session,
                env=env,
                account_id=account_id,
                instruments=instruments,
                headers=headers,
                out_queue=queue,
                last_price=last_price,
                stop_event=stop_event,
                snapshot=not args.no_stream_snapshot,
            )
        )

        tasks = [writer, stream]

        # Start 1-second snapshot derivation
        if not args.disable_1s_snapshot:
            snap = asyncio.create_task(pricing_1s_snapshot_task(instruments, last_price, queue, stop_event))
            tasks.append(snap)

        # FXLabs polling tasks (optional)
        if args.labs_orderbook:
            ob = asyncio.create_task(
                labs_orderbook_poll_task(
                    session=session,
                    env=env,
                    instruments=instruments,
                    headers=headers,
                    out_queue=queue,
                    stop_event=stop_event,
                    poll_interval_s=args.labs_orderbook_poll,
                    period_s=args.labs_orderbook_period,
                    log_full_response=args.labs_log_full,
                )
            )
            tasks.append(ob)

        if args.labs_position_ratios:
            pr = asyncio.create_task(
                labs_position_ratios_poll_task(
                    session=session,
                    env=env,
                    instruments=instruments,
                    headers=headers,
                    out_queue=queue,
                    stop_event=stop_event,
                    poll_interval_s=args.labs_position_poll,
                    period_s=args.labs_position_period,
                    log_full_response=args.labs_log_full,
                )
            )
            tasks.append(pr)

        # V20 Polling
        if args.v20_books:
            v20_ob = asyncio.create_task(
                v20_orderbook_poll_task(
                    session=session,
                    env=env,
                    instruments=instruments,
                    headers=headers,
                    out_queue=queue,
                    stop_event=stop_event,
                    poll_interval_s=args.v20_poll,
                )
            )
            v20_pb = asyncio.create_task(
                v20_positionbook_poll_task(
                    session=session,
                    env=env,
                    instruments=instruments,
                    headers=headers,
                    out_queue=queue,
                    stop_event=stop_event,
                    poll_interval_s=args.v20_poll,
                )
            )
            tasks.extend([v20_ob, v20_pb])

        # Add a stats logger to show activity in the log file every 60s
        async def stats_logger():
            while not stop_event.is_set():
                await asyncio.sleep(60)
                logging.info("Collector status: Connected. Queue size: %d. Last prices processed for %s", 
                             queue.qsize(), ", ".join(last_price.keys()))
        
        stats_task = asyncio.create_task(stats_logger())
        tasks.append(stats_task)

        # Wait until stop requested
        while not stop_event.is_set():
            await asyncio.sleep(0.25)

        # Stop: cancel streaming & pollers, then drain writer
        for t in tasks:
            if t is writer:
                continue
            t.cancel()

        await asyncio.gather(*[t for t in tasks if t is not writer], return_exceptions=True)

        # Let writer flush remaining queue
        await queue.join()
        stop_event.set()
        await asyncio.gather(writer, return_exceptions=True)

    logging.info("Done. Logs in: %s", str(run_dir))
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
