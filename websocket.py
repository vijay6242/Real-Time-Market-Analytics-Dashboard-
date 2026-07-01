# websocket.py — Real-Time Market Data Streaming Hub
# Adapted from your websocket-4.py for Streamlit integration.
#
# Architecture:
#   • Singleton WebSocket — one connection for all subscriptions
#   • Runs in a background daemon thread (asyncio event loop)
#   • Live ticks stored in a shared thread-safe dict (tick_store)
#   • Streamlit polls tick_store via st.session_state every N seconds
#   • Protobuf decoding via MarketDataFeedV3_pb2 (if available),
#     falls back to JSON mode for testing without the .proto file
#
# Data flow:
#   Upstox WS → _decode() → tick_store[instrument_key] = {ltp, change, ...}
#   Streamlit → polls tick_store → updates charts/metrics

import asyncio
import json
import ssl
import time
import threading
import traceback
from datetime import datetime
from collections import deque
from typing import Callable, Optional

import aiohttp
import websockets

from logger import get_logger

logger = get_logger(__name__)

# ── Upstox endpoints ──────────────────────────────────────
_AUTH_URL      = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
_BACKOFF_BASE  = 2
_BACKOFF_CAP   = 60
_MAX_RETRIES   = 10
_RECV_TIMEOUT  = 60
_PING_INTERVAL = 30

# ── Protobuf import (graceful fallback) ───────────────────
try:
    import MarketDataFeedV3_pb2 as pb
    _PROTO_AVAILABLE = True
    logger.info("[WS] Protobuf decoder loaded ✅")
except ImportError:
    _PROTO_AVAILABLE = False
    logger.warning("[WS] MarketDataFeedV3_pb2 not found — JSON fallback mode")


# ─────────────────────────────────────────────────────────
# Shared Tick Store (thread-safe for Streamlit polling)
# ─────────────────────────────────────────────────────────

class TickStore:
    """
    Central in-memory store for live market ticks.
    Written by the WebSocket thread, read by Streamlit's main thread.

    Structure per key:
        tick_store[instrument_key] = {
            'ltp':       float,
            'prev_ltp':  float,
            'change':    float,   # absolute
            'change_pct':float,   # percent
            'timestamp': str,     # HH:MM:SS
            'history':   deque([ltp, ...], maxlen=300)
        }
    """
    def __init__(self):
        self._data: dict = {}
        self._lock = threading.Lock()

    def update(self, instrument_key: str, ltp: float):
        with self._lock:
            if instrument_key not in self._data:
                self._data[instrument_key] = {
                    'ltp':        ltp,
                    'prev_ltp':   ltp,
                    'change':     0.0,
                    'change_pct': 0.0,
                    'timestamp':  datetime.now().strftime("%H:%M:%S"),
                    'history':    deque([ltp], maxlen=300),
                }
            else:
                prev = self._data[instrument_key]['ltp']
                self._data[instrument_key].update({
                    'prev_ltp':   prev,
                    'ltp':        ltp,
                    'change':     round(ltp - prev, 2),
                    'change_pct': round((ltp - prev) / prev * 100, 3) if prev else 0,
                    'timestamp':  datetime.now().strftime("%H:%M:%S"),
                })
                self._data[instrument_key]['history'].append(ltp)

    def get(self, instrument_key: str) -> Optional[dict]:
        with self._lock:
            return dict(self._data.get(instrument_key, {}))

    def get_all(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def clear(self, instrument_key: str = None):
        with self._lock:
            if instrument_key:
                self._data.pop(instrument_key, None)
            else:
                self._data.clear()


# Global shared tick store — imported by app.py
tick_store = TickStore()


# ─────────────────────────────────────────────────────────
# WebSocket Hub (Singleton)
# ─────────────────────────────────────────────────────────

class UpstoxWebSocket:
    """
    Singleton WebSocket hub. Manages one persistent connection to Upstox.
    Runs entirely in a background daemon thread.

    Usage from Streamlit app.py:
        from websocket import ws_hub, tick_store
        ws_hub.add_subscription("NSE_INDEX|Nifty 50")
        ws_hub.start(access_token)

        # In Streamlit render loop:
        tick = tick_store.get("NSE_INDEX|Nifty 50")
        if tick:
            st.metric("NIFTY 50", tick['ltp'], tick['change_pct'])
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.access_token: str = ""
        self.is_running: bool  = False

        self.instrument_keys: set = set()
        self._subscriptions_dirty: bool = False

        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._ws_conn = None
        self._main_task: asyncio.Task = None

        # Status for UI
        self.status: str = "Disconnected"
        self.status_callback: Optional[Callable] = None

        # Tick timing
        self.last_tick_time: float = 0.0
        self.tick_count: int = 0

        self._initialized = True
        logger.info("[WS] UpstoxWebSocket singleton ready.")

    # ── Public API ────────────────────────────────────────

    def start(self, access_token: str):
        """Start the WebSocket stream in a background thread."""
        if self.is_running:
            logger.info("[WS] Already running — adding subscriptions only.")
            if self._subscriptions_dirty and self._loop:
                asyncio.run_coroutine_threadsafe(self._send_subscription(), self._loop)
            return

        self.access_token = access_token
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="UpstoxWS"
        )
        self._thread.start()
        logger.info("[WS] Background thread started.")

    def stop(self):
        """Gracefully stop the WebSocket."""
        self.is_running = False
        if self._loop and self._ws_conn:
            asyncio.run_coroutine_threadsafe(
                self._ws_conn.close(), self._loop
            )
        self._notify_status("Disconnected")
        logger.info("[WS] Stop requested.")

    def add_subscription(self, instrument_key: str):
        """Add a single instrument key to the live subscription."""
        if instrument_key not in self.instrument_keys:
            self.instrument_keys.add(instrument_key)
            self._subscriptions_dirty = True
            logger.info(f"[WS] Queued subscription: {instrument_key}")
            if self.is_running and self._loop:
                asyncio.run_coroutine_threadsafe(self._send_subscription(), self._loop)

    def remove_subscription(self, instrument_key: str):
        """Remove an instrument from subscriptions."""
        self.instrument_keys.discard(instrument_key)
        tick_store.clear(instrument_key)
        if self.is_running and self._loop and self.instrument_keys:
            asyncio.run_coroutine_threadsafe(self._send_subscription(), self._loop)

    def add_subscriptions(self, keys: list):
        """Bulk add instrument keys."""
        new = [k for k in keys if k not in self.instrument_keys]
        if new:
            self.instrument_keys.update(new)
            self._subscriptions_dirty = True
            if self.is_running and self._loop:
                asyncio.run_coroutine_threadsafe(self._send_subscription(), self._loop)

    @property
    def subscribed_count(self) -> int:
        return len(self.instrument_keys)

    @property
    def diagnostics(self) -> dict:
        return {
            'status':          self.status,
            'is_running':      self.is_running,
            'subscribed_keys': self.subscribed_count,
            'tick_count':      self.tick_count,
            'last_tick':       datetime.fromtimestamp(self.last_tick_time).strftime("%H:%M:%S")
                               if self.last_tick_time > 0 else "—",
        }

    # ── Internal: Event Loop Thread ───────────────────────

    def _run_loop(self):
        """Runs the asyncio event loop in the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._connect_loop())
        finally:
            loop.close()
            self._loop = None
            self.is_running = False

    # ── Internal: Connection Loop ─────────────────────────

    async def _connect_loop(self):
        """Outer reconnect loop with exponential backoff (from your websocket-4.py)."""
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode    = ssl.CERT_NONE
        consecutive_failures   = 0

        while True:
            heartbeat_task = None
            try:
                self._notify_status("Authorizing")
                uri = await self._authorize()

                self._notify_status("Connecting")
                async with websockets.connect(
                    uri,
                    ssl=ssl_ctx,
                    ping_interval=_PING_INTERVAL,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws_conn = ws
                    self.is_running = True
                    consecutive_failures = 0
                    self.last_tick_time = time.monotonic()
                    self._notify_status("Connected ✅")
                    logger.info("[WS] Connected to Upstox feed.")

                    await self._send_subscription()
                    heartbeat_task = asyncio.ensure_future(self._heartbeat_monitor())

                    # ── Receive loop ──────────────────────
                    while self.is_running:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=_RECV_TIMEOUT
                            )
                            tick = self._decode(raw)
                            if tick:
                                for key, ltp in tick.items():
                                    tick_store.update(key, ltp)
                                self.last_tick_time = time.monotonic()
                                self.tick_count += len(tick)

                        except asyncio.TimeoutError:
                            try:
                                await ws.ping()
                            except Exception:
                                logger.warning("[WS] Ping failed — reconnecting.")
                                break

                        except websockets.exceptions.ConnectionClosed as cc:
                            logger.warning(f"[WS] Connection closed: {cc}")
                            self._notify_status("Reconnecting...")
                            break

                        except asyncio.CancelledError:
                            raise

                        except Exception as e:
                            logger.error(f"[WS] Recv error: {e}")
                            await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                logger.info("[WS] Cancelled — shutting down.")
                break

            except Exception as e:
                consecutive_failures += 1
                wait = min(_BACKOFF_BASE ** consecutive_failures, _BACKOFF_CAP)
                if consecutive_failures >= _MAX_RETRIES:
                    logger.error("[WS] Max retries reached — giving up.")
                    self._notify_status("Error ❌")
                    break
                logger.warning(f"[WS] Attempt {consecutive_failures} failed: {e}. Retry in {wait}s")
                self._notify_status(f"Retry {consecutive_failures}/{_MAX_RETRIES}")
                await asyncio.sleep(wait)

            finally:
                if heartbeat_task and not heartbeat_task.done():
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except Exception:
                        pass
                self._ws_conn   = None
                self.is_running = False

        self._notify_status("Disconnected")

    # ── Internal: Authorization (from websocket-4.py) ────

    async def _authorize(self) -> str:
        headers = {
            'Accept':        'application/json',
            'Authorization': f'Bearer {self.access_token}',
        }
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            for attempt in range(5):
                try:
                    async with session.get(
                        _AUTH_URL, timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        uri  = data.get('data', {}).get('authorized_redirect_uri')
                        if not uri:
                            raise ValueError("No authorized_redirect_uri in response.")
                        logger.info("[WS] Authorization OK.")
                        return uri
                except Exception as e:
                    wait = min(_BACKOFF_BASE ** attempt, _BACKOFF_CAP)
                    logger.warning(f"[WS] Auth attempt {attempt+1} failed: {e}. Wait {wait}s")
                    if attempt < 4:
                        await asyncio.sleep(wait)
        raise Exception("[WS] Authorization failed after all retries.")

    # ── Internal: Subscription Sender (from websocket-4.py) ──

    async def _send_subscription(self):
        if not self._ws_conn or not self.instrument_keys:
            return
        try:
            msg = {
                "guid":   f"sub_{time.time():.0f}",
                "method": "sub",
                "data": {
                    "mode":           "ltpc",
                    "instrumentKeys": list(self.instrument_keys),
                },
            }
            # Binary frame required by Upstox V3 Protobuf
            await self._ws_conn.send(json.dumps(msg).encode('utf-8'))
            self._subscriptions_dirty = False
            logger.info(f"[WS] Subscribed {len(self.instrument_keys)} keys in ltpc mode.")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("[WS] Cannot subscribe — connection closed.")
        except Exception as e:
            logger.warning(f"[WS] Subscription error: {e}")

    # ── Internal: Protobuf Decoder (from websocket-4.py) ──

    def _decode(self, raw_bytes: bytes) -> dict:
        """Decode raw Protobuf or JSON bytes → {instrument_key: ltp}"""
        if _PROTO_AVAILABLE:
            try:
                feed_response = pb.FeedResponse()
                feed_response.ParseFromString(raw_bytes)
                result = {}
                for key, feed in feed_response.feeds.items():
                    ltp = 0.0
                    if feed.HasField('ltpc'):
                        ltp = feed.ltpc.ltp
                    elif feed.HasField('ff'):
                        ff = feed.ff
                        if ff.HasField('marketFF'):
                            ltp = ff.marketFF.ltpc.ltp
                        elif ff.HasField('indexFF'):
                            ltp = ff.indexFF.ltpc.ltp
                    if ltp > 0:
                        result[key] = ltp
                return result
            except Exception as e:
                logger.debug(f"[WS] Proto decode error: {e}")
                return {}
        else:
            # JSON fallback
            try:
                data = json.loads(raw_bytes)
                result = {}
                feeds = data.get('feeds', {})
                for key, feed in feeds.items():
                    ltpc = feed.get('ltpc', {})
                    ltp  = float(ltpc.get('ltp', 0))
                    if ltp > 0:
                        result[key] = ltp
                return result
            except Exception:
                return {}

    # ── Internal: Heartbeat Monitor (from websocket-4.py) ──

    async def _heartbeat_monitor(self):
        _INTERVAL = 5
        _TIMEOUT  = 30
        try:
            while True:
                await asyncio.sleep(_INTERVAL)
                silence = time.monotonic() - self.last_tick_time
                if self.last_tick_time > 0 and silence > _TIMEOUT:
                    logger.warning(f"[WS] No tick for {silence:.0f}s — forcing reconnect.")
                    self._notify_status("Reconnecting (heartbeat)...")
                    if self._ws_conn:
                        try:
                            await self._ws_conn.close()
                        except Exception:
                            pass
                    break
        except asyncio.CancelledError:
            pass

    def _notify_status(self, status: str):
        self.status = status
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception:
                pass
        logger.debug(f"[WS] Status → {status}")


# ── Global singleton hub (imported by app.py) ─────────────
ws_hub = UpstoxWebSocket()