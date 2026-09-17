"""Recorder end-to-end against a local fake Nord Pool market-data server (SockJS + STOMP)."""
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
websockets = pytest.importorskip("websockets")
from v4_1_intraday.collectors import nordpool_id as npid  # noqa: E402


def _frame(cmd, headers, body=""):
    return npid.stomp_frame(cmd, headers, body)


async def _fake_server(ws):
    await ws.send("o")
    subs = {}
    async for raw in ws:
        for fr in json.loads(raw):
            p = npid.parse_frame(fr)
            if not p:
                continue
            cmd, h, _ = p
            if cmd == "CONNECT":
                assert h["X-AUTH-TOKEN"] == "tok"
                await ws.send("a" + json.dumps([_frame("CONNECTED", {"version": "1.2"})]))
            elif cmd == "SUBSCRIBE":
                d = h["destination"]
                subs[d] = h["id"]
                if d.endswith("/deliveryAreas"):
                    body = [{"deliveryAreaId": 3, "eicCode": "10YDK-1--------W", "areaCode": "DK1"},
                            {"deliveryAreaId": 99, "eicCode": "10YFI-1--------U", "areaCode": "FI"}]
                elif "/publicStatistics/3" in d:
                    body = [{"contractId": "C1", "deliveryAreaId": 3, "lastPrice": 12345, "vwap": 12000,
                             "lastQuantity": 1000, "highestPrice": 13000, "lowestPrice": 11000,
                             "turnover": 50000, "dayAheadPrice": 11500, "updatedAt": "2026-09-17T10:00:00Z"}]
                elif d.endswith("/ticker"):
                    body = [{"tradeId": "T1", "tradeTime": "2026-09-17T10:00:01Z", "state": "ACTI",
                             "legs": [{"contractId": "C1", "deliveryAreaId": 3, "side": "BUY", "unitPrice": 12100,
                                       "quantity": 2000, "aggressor": True},
                                      {"contractId": "C1", "deliveryAreaId": 99, "side": "SELL",
                                       "unitPrice": 12100, "quantity": 2000, "aggressor": False}]}]
                elif "/localview/3" in d:
                    body = [{"contractId": "C1", "deliveryAreaId": 3, "revisionNo": 1,
                             "buyOrders": [{"orderId": "b1", "price": 11900, "qty": 3000}],
                             "sellOrders": [{"orderId": "s1", "price": 12200, "qty": 1000}]}]
                else:
                    continue
                msg = _frame("MESSAGE", {"destination": d, "subscription": h["id"], "x-nps-snapshot": "true",
                                         "x-nps-sent-at": "1789639200000"}, json.dumps(body))
                await ws.send("a" + json.dumps([msg]))


def test_recorder_sockjs_end_to_end(tmp_path, monkeypatch):
    async def main():
        server = await websockets.serve(_fake_server, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        real_connect = websockets.connect
        seen_urls = []

        def fake_connect(url, **kw):
            seen_urls.append(url)
            return real_connect(f"ws://127.0.0.1:{port}", **kw)

        monkeypatch.setattr(websockets, "connect", fake_connect)
        monkeypatch.setattr(npid, "get_token", lambda c=None: ("tok", "IDAPI_TEST"))
        c = dict(npid.DEFAULTS, flush_seconds=0.5, heartbeat_ms=200)
        router = npid.Router(c)
        await npid._session(c, router, tmp_path, 3.0, True)
        server.close()
        await server.wait_closed()
        return seen_urls

    urls = asyncio.run(main())
    assert "/user/" in urls[0] and urls[0].endswith("/websocket")
    stats = npid.load("stats", root=tmp_path)
    trades = npid.load("trades", root=tmp_path)
    book = npid.load("book", root=tmp_path)
    areas = npid.load("areas", root=tmp_path)
    assert (areas["dk"] == "DK1").any()
    assert len(stats) == 1 and stats["vwap"].iloc[0] == 12000
    assert len(trades) == 1 and trades["area_id"].iloc[0] == 3          # FI leg filtered out
    assert book["best_bid"].iloc[0] == 11900 and book["best_ask"].iloc[0] == 12200


def test_sockjs_unpack():
    assert npid.sockjs_unpack("o") == [] and npid.sockjs_unpack("h") == []
    assert npid.sockjs_unpack('a["X\\n\\n\\u0000"]') == ["X\n\n\x00"]
    assert npid.sockjs_unpack('c[3000,"Go away!"]') is None
