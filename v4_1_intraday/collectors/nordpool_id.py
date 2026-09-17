"""Nord Pool Intraday API (read-only) recorder: live public market data -> parquet files.

The read-only Intraday account gives live data only (no history), so this recorder must run
24/7 from now on. Everything is stored raw, stamped with the local receive time (UTC), so the
feature builder can use exactly what was known at each decision time.

Protocol (Nord Pool public-intraday-api example):
  * SSO token: POST <sso_url>  Basic base64(client_id:client_secret),
    form grant_type=password, scope=global, username, password
  * WebSocket, SockJS framing as in Nord Pool's official client:
    wss://<market_data_host>:443/user/<3-digit server id>/<session id>/websocket
    server sends "o" (open), "h" (heartbeat), a["<stomp frame>", ...]; client sends ["<stomp frame>"].
    Fallback: plain STOMP on /user/websocket or /user. STOMP 1.2, CONNECT header X-AUTH-TOKEN
  * Subscriptions: /user/<user>/v1/streaming/deliveryAreas
                   /user/<user>/v1/conflated/contracts
                   /user/<user>/v1/conflated/publicStatistics/<areaId>
                   /user/<user>/v1/streaming/ticker
                   /user/<user>/v1/conflated/localview/<areaId>
Prices arrive in cents (EUR/MWh * 100); quantity scale is configurable (intraday_api.qty_divisor)
and is checked by `train_v4_1.py probe-nordpool`.

Output: <data_dir>/<table>/date=YYYY-MM-DD/part-<HHMMSS>-<n>.parquet
  tables: areas, contracts, stats, trades, book
"""
from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import time
import zlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from .. import settings as S
from .common import data_dir, secret, utc_naive

log = logging.getLogger("nurex41id.nordpool")

DEFAULTS = {
    "sso_url": "https://sso.nordpoolgroup.com/connect/token",
    "client_id": "client_intraday_api",
    "client_secret": "1xB9Ik1xsEu2nbwVa1BR",       # public client secret from Nord Pool's example
    "market_data_host": "intraday-pmd-api-ws-nordpool.nordpoolgroup.com",
    "ws_paths": ["sockjs", "/user/websocket", "/user"],
    "heartbeat_ms": 5000,
    "flush_seconds": 60,
    "areas_eic": {"DK1": "10YDK-1--------W", "DK2": "10YDK-2--------M"},
    "area_ids": {},                                  # fallback if deliveryAreas is not received
    "price_divisor": 100.0,
    "qty_divisor": 1000.0,
    "book_depth_eur": 5.0,
}


def cfg_api() -> dict:
    c = dict(DEFAULTS)
    c.update(S.load().raw.get("intraday_api", {}) or {})
    for k, env in (("sso_url", "NORDPOOL_ID_SSO_URL"), ("market_data_host", "NORDPOOL_ID_MD_HOST"),
                   ("client_secret", "NORDPOOL_ID_CLIENT_SECRET"), ("client_id", "NORDPOOL_ID_CLIENT_ID")):
        v = secret(env)
        if v:
            c[k] = v
    return c


# ----------------------------------------------------------------------------- auth
def get_token(c: dict | None = None) -> tuple[str, str]:
    """Returns (token, username that worked). Tries NORDPOOL_ID_USERNAME, then NORDPOOL_ID_EMAIL."""
    c = c or cfg_api()
    pwd = secret("NORDPOOL_ID_PASSWORD")
    users = [u for u in (secret("NORDPOOL_ID_USERNAME"), secret("NORDPOOL_ID_EMAIL")) if u]
    if not pwd or not users:
        raise RuntimeError("NORDPOOL_ID_USERNAME / NORDPOOL_ID_PASSWORD missing in Nurex_V4_2/.env")
    basic = base64.b64encode(f"{c['client_id']}:{c['client_secret']}".encode()).decode()
    errors = []
    for u in users:
        r = requests.post(c["sso_url"], data={"grant_type": "password", "scope": "global", "username": u,
                                              "password": pwd},
                          headers={"Authorization": f"Basic {basic}",
                                   "Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
        if r.status_code == 200 and "access_token" in r.json():
            return r.json()["access_token"], u
        errors.append(f"{u}: HTTP {r.status_code} {r.text[:150]}")
    raise RuntimeError("Nord Pool SSO login failed: " + " | ".join(errors))


def token_expiry(token: str) -> float:
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        return float(json.loads(base64.urlsafe_b64decode(part))["exp"])
    except Exception:
        return time.time() + 3000


# ----------------------------------------------------------------------------- STOMP
def stomp_frame(command: str, headers: dict, body: str = "") -> str:
    h = "".join(f"{k}:{v}\n" for k, v in headers.items())
    return f"{command}\n{h}\n{body}\x00"


def parse_frame(raw) -> tuple[str, dict, str] | None:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    raw = raw.lstrip("\r\n")
    if not raw:
        return None                                   # heartbeat
    head, _, body = raw.partition("\n\n")
    lines = head.split("\n")
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers.setdefault(k.strip(), v.strip())
    return lines[0].strip(), headers, body.rstrip("\x00\r\n")


def decode_body(body: str):
    try:
        return json.loads(body)
    except Exception:
        pass
    b = body.encode("latin-1", errors="ignore")
    for f in (gzip.decompress, lambda x: zlib.decompress(x, 15 + 32), base64.b64decode):
        try:
            out = f(b)
            return json.loads(out if isinstance(out, (bytes, str)) else out)
        except Exception:
            continue
    return None


def sockjs_unpack(msg) -> list[str] | None:
    """SockJS server message -> list of STOMP frame strings ([] for open/heartbeat, None for close)."""
    if isinstance(msg, (bytes, bytearray)):
        msg = msg.decode("utf-8", errors="replace")
    if not msg or msg in ("o", "h"):
        return []
    if msg[0] == "a":
        return [str(x) for x in json.loads(msg[1:])]
    if msg[0] == "c":
        return None
    return [msg]                                      # not SockJS framed


class Transport:
    """Hides the difference between SockJS-framed and plain STOMP websockets."""

    def __init__(self, ws, sockjs: bool):
        self.ws, self.sockjs, self.buffer = ws, sockjs, []

    async def send(self, frame: str):
        await self.ws.send(json.dumps([frame]) if self.sockjs else frame)

    async def recv(self, timeout: float):
        """Next STOMP frame string, '' on heartbeat/timeout; raises ConnectionError when closed."""
        if self.buffer:
            return self.buffer.pop(0)
        try:
            msg = await asyncio.wait_for(self.ws.recv(), timeout)
        except asyncio.TimeoutError:
            return ""
        if not self.sockjs:
            return msg
        frames = sockjs_unpack(msg)
        if frames is None:
            raise ConnectionError(f"server closed the session: {msg[:200]}")
        self.buffer.extend(frames)
        return self.buffer.pop(0) if self.buffer else ""


# ----------------------------------------------------------------------------- message -> rows
class Book:
    def __init__(self):
        self.orders = {}          # orderId -> (side, price, qty)

    def apply(self, row: dict, snapshot: bool):
        if snapshot:
            self.orders = {}
        for side, key in (("B", "buyOrders"), ("S", "sellOrders")):
            for o in row.get(key) or []:
                oid = o.get("orderId")
                if o.get("deleted"):
                    self.orders.pop(oid, None)
                elif oid is not None:
                    self.orders[oid] = (side, o.get("price"), o.get("qty"))

    def summary(self, depth_cents: float) -> dict:
        bids = [(p, q) for s, p, q in self.orders.values() if s == "B" and p is not None]
        asks = [(p, q) for s, p, q in self.orders.values() if s == "S" and p is not None]
        bb = max((p for p, _ in bids), default=None)
        ba = min((p for p, _ in asks), default=None)
        return {"best_bid": bb, "best_ask": ba,
                "bid_qty_best": sum(q or 0 for p, q in bids if p == bb) if bb is not None else None,
                "ask_qty_best": sum(q or 0 for p, q in asks if p == ba) if ba is not None else None,
                "bid_qty_depth": sum(q or 0 for p, q in bids if bb is not None and p >= bb - depth_cents),
                "ask_qty_depth": sum(q or 0 for p, q in asks if ba is not None and p <= ba + depth_cents),
                "n_bid": len(bids), "n_ask": len(asks)}


class Router:
    """Turns STOMP MESSAGE payloads into flat rows per table."""

    def __init__(self, c: dict):
        self.c = c
        self.area_ids: dict[int, str] = {}           # nord pool area id -> DK1/DK2
        self.books = defaultdict(Book)
        self.rows = defaultdict(list)

    def handle(self, dest: str, headers: dict, payload, recv: pd.Timestamp):
        items = payload if isinstance(payload, list) else [payload]
        snap = str(headers.get("x-nps-snapshot", "")).lower() == "true"
        sent = headers.get("x-nps-sent-at")
        if "/deliveryAreas" in dest:
            eic2dk = {v: k for k, v in self.c["areas_eic"].items()}
            for it in items:
                dk = eic2dk.get(it.get("eicCode"))
                if dk:
                    self.area_ids[int(it["deliveryAreaId"])] = dk
                self.rows["areas"].append({"recv_utc": recv, "area_id": it.get("deliveryAreaId"),
                                           "eic": it.get("eicCode"), "area_code": it.get("areaCode"),
                                           "dk": dk})
        elif "/contracts" in dest:
            for it in items:
                self.rows["contracts"].append({
                    "recv_utc": recv, "contract_id": it.get("contractId"), "name": it.get("contractName"),
                    "product_type": str(it.get("productType")), "product_id": it.get("productId"),
                    "dlvry_start": it.get("dlvryStart"), "dlvry_end": it.get("dlvryEnd"),
                    "duration_s": it.get("durationSeconds"), "deleted": it.get("deleted"),
                    "area_ids": json.dumps([a.get("dlvryAreaId", a.get("deliveryAreaId"))
                                            for a in (it.get("dlvryAreaState") or [])])})
        elif "/publicStatistics" in dest:
            for it in items:
                self.rows["stats"].append({
                    "recv_utc": recv, "sent_at": sent, "contract_id": it.get("contractId"),
                    "area_id": it.get("deliveryAreaId"), "last_price": it.get("lastPrice"),
                    "last_qty": it.get("lastQuantity"), "last_trade_time": it.get("lastTradeTime"),
                    "high": it.get("highestPrice"), "low": it.get("lowestPrice"), "vwap": it.get("vwap"),
                    "turnover": it.get("turnover"), "da_price": it.get("dayAheadPrice"),
                    "deleted": it.get("deleted"), "updated_at": it.get("updatedAt")})
        elif "/ticker" in dest:
            for it in items:
                for leg in it.get("legs") or []:
                    aid = leg.get("deliveryAreaId")
                    if self.area_ids and aid not in self.area_ids:
                        continue
                    self.rows["trades"].append({
                        "recv_utc": recv, "trade_id": it.get("tradeId"),
                        "trade_time": it.get("tradeTime") or it.get("eventTime") or it.get("updatedAt"),
                        "state": str(it.get("state", "")), "deleted": it.get("deleted"),
                        "contract_id": leg.get("contractId"), "area_id": aid, "side": str(leg.get("side")),
                        "price": leg.get("unitPrice"), "qty": leg.get("quantity"),
                        "aggressor": leg.get("aggressor")})
        elif "/localview" in dest:
            depth = float(self.c["book_depth_eur"]) * float(self.c["price_divisor"])
            for it in items:
                key = (it.get("contractId"), it.get("deliveryAreaId"))
                b = self.books[key]
                b.apply(it, snap)
                self.rows["book"].append({"recv_utc": recv, "contract_id": key[0], "area_id": key[1],
                                          "revision": it.get("revisionNo"), **b.summary(depth)})

    def drain(self) -> dict:
        out, self.rows = self.rows, defaultdict(list)
        return out


def write_rows(rows: dict, root: Path) -> int:
    n = 0
    stamp = pd.Timestamp.now(tz="UTC")
    for table, lst in rows.items():
        if not lst:
            continue
        df = pd.DataFrame(lst)
        for c in ("sent_at", "last_trade_time", "updated_at", "dlvry_start", "dlvry_end", "trade_time"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)
        if "recv_utc" in df.columns:
            df["recv_utc"] = pd.to_datetime(df["recv_utc"], utc=True).dt.tz_localize(None)
        d = root / table / f"date={stamp:%Y-%m-%d}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / f"part-{stamp:%H%M%S}-{len(df)}.parquet", index=False)
        n += len(df)
    return n


# ----------------------------------------------------------------------------- recorder
async def _session(c: dict, router: Router, root: Path, stop_after: float | None, subscribe_localview: bool):
    import websockets
    token, user = get_token(c)
    exp = token_expiry(token)
    last_err = None
    tr = None
    for path in c["ws_paths"]:
        if path == "sockjs":
            server_id = f"{__import__('random').randint(0, 999):03d}"
            url = f"wss://{c['market_data_host']}:443/user/{server_id}/{__import__('uuid').uuid4().hex}/websocket"
        else:
            url = f"wss://{c['market_data_host']}:443{path}"
        try:
            ws = await websockets.connect(url, max_size=None, open_timeout=20, ping_interval=None)
        except Exception as e:
            last_err = e
            log.warning("connect %s failed: %s", url, e)
            continue
        cand = Transport(ws, sockjs=(path == "sockjs"))
        try:
            if cand.sockjs:                           # wait for the SockJS open frame "o"
                first = await asyncio.wait_for(ws.recv(), 20)
                if first != "o":
                    raise RuntimeError(f"unexpected SockJS greeting {str(first)[:80]}")
            await cand.send(stomp_frame("CONNECT", {"accept-version": "1.2,1.1,1.0", "X-AUTH-TOKEN": token,
                                                    "heart-beat": f"0,{int(c['heartbeat_ms'])}"}))
            fr = None
            t0 = time.time()
            while time.time() - t0 < 20:
                raw = await cand.recv(5)
                fr = parse_frame(raw) if raw else None
                if fr:
                    break
            if not fr or fr[0] != "CONNECTED":
                raise RuntimeError(f"STOMP connect refused: {fr}")
            tr = cand
            log.info("connected to %s (%s) as %s", c["market_data_host"], "SockJS" if cand.sockjs else "STOMP", user)
            break
        except Exception as e:
            last_err = e
            log.warning("handshake on %s failed: %s", url, e)
            await ws.close()
    if tr is None:
        raise RuntimeError(f"cannot open Nord Pool market data session: {last_err}")
    hb = int(c["heartbeat_ms"])
    async with tr.ws:
        sub_id = 0

        async def sub(dest):
            nonlocal sub_id
            sub_id += 1
            await tr.send(stomp_frame("SUBSCRIBE", {"destination": dest, "id": f"sub-{sub_id}"}))

        base = f"/user/{user}/v1"
        await sub(f"{base}/streaming/deliveryAreas")
        await sub(f"{base}/conflated/contracts")
        await sub(f"{base}/streaming/ticker")

        async def heartbeat():
            while True:
                await asyncio.sleep(hb / 1000.0)
                await tr.send("\n")

        hb_task = asyncio.create_task(heartbeat())
        started, last_flush, area_subscribed = time.time(), time.time(), False
        try:
            while True:
                now = time.time()
                if stop_after and now - started > stop_after:
                    break
                if now > exp - 300:
                    log.info("token expiring - reconnecting")
                    break
                if not area_subscribed and (router.area_ids or now - started > 20):
                    ids = router.area_ids or {int(v): k for k, v in (c.get("area_ids") or {}).items()}
                    if not ids:
                        log.error("DK delivery area ids unknown (no deliveryAreas message) - set intraday_api.area_ids")
                    for aid in ids:
                        await sub(f"{base}/conflated/publicStatistics/{aid}")
                        if subscribe_localview:
                            await sub(f"{base}/conflated/localview/{aid}")
                    area_subscribed = True
                    log.info("subscribed to statistics/order book for areas %s", ids)
                msg = await tr.recv(5)
                if msg:
                    fr = parse_frame(msg)
                    if fr and fr[0] == "MESSAGE":
                        payload = decode_body(fr[2])
                        if payload is not None:
                            router.handle(fr[1].get("destination", ""), fr[1], payload, pd.Timestamp.now(tz="UTC"))
                    elif fr and fr[0] == "ERROR":
                        log.error("STOMP error: %s %s", fr[1].get("message"), fr[2][:200])
                if time.time() - last_flush > float(c["flush_seconds"]):
                    n = write_rows(router.drain(), root)
                    last_flush = time.time()
                    log.info("flushed %d rows", n)
        finally:
            hb_task.cancel()
            write_rows(router.drain(), root)


def record(stop_after: float | None = None, subscribe_localview: bool = True) -> None:
    """Run forever (or `stop_after` seconds), reconnecting on errors and before token expiry."""
    c = cfg_api()
    root = data_dir()
    router = Router(c)
    started = time.time()
    delay = 5
    while True:
        remaining = None if stop_after is None else stop_after - (time.time() - started)
        if remaining is not None and remaining <= 0:
            return
        try:
            asyncio.run(_session(c, router, root, remaining, subscribe_localview))
            delay = 5
        except KeyboardInterrupt:
            return
        except Exception as e:
            log.error("recorder session ended: %s - retry in %ds", e, delay)
            time.sleep(delay)
            delay = min(delay * 2, 300)


# ----------------------------------------------------------------------------- reader
def load(table: str, start=None, end=None, root: Path | None = None) -> pd.DataFrame:
    root = root or data_dir()
    d = root / table
    if not d.exists():
        return pd.DataFrame()
    files = []
    for p in sorted(d.glob("date=*")):
        day = pd.Timestamp(p.name.split("=", 1)[1])
        if start is not None and day < pd.Timestamp(start).normalize() - pd.Timedelta(days=1):
            continue
        if end is not None and day > pd.Timestamp(end).normalize() + pd.Timedelta(days=1):
            continue
        files += sorted(p.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def coverage(root: Path | None = None) -> pd.DataFrame:
    rows = []
    for t in ("areas", "contracts", "stats", "trades", "book"):
        df = load(t, root=root)
        rows.append({"table": t, "rows": len(df),
                     "first": df["recv_utc"].min() if len(df) else None,
                     "last": df["recv_utc"].max() if len(df) else None})
    return pd.DataFrame(rows)
