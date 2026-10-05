"""Imbalance forecast panel (2026-10-05): one self-contained HTML/SVG block for the dashboard.

Dark, high-contrast layout in the spirit of commercial imbalance forecasters: per quarter a
10-90% range bar, median, expected value, day-ahead (spot) price, a "clock" ring with the
down / none / up probabilities, the most likely direction, certainty, our decision, and - once
settled - the actual outcome and P&L. Hover any quarter for the full numbers.
Display only; input is the frame built in the dashboard tab (authentic model + journal data).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

C = {"bg": "#1b1e2b", "panel": "#232738", "grid": "#343a50", "axis": "#aab3c5", "text": "#e9edf5",
     "band": "#5b8fc0", "band_edge": "#8db8e0", "median": "#ffffff", "point": "#ff4d5e", "spot": "#ffd166",
     "actual": "#00e5a8", "down": "#ff8a4c", "none": "#4cd08a", "up": "#4aa8ff", "amb": "#8a93a6",
     "high": "#2ee59d", "medium": "#ffd166", "low": "#ff7a7a", "past": "rgba(0,0,0,0.28)", "now": "#ffffff"}
DIR_COL = {"Down": C["down"], "None": C["none"], "Up": C["up"], "-": C["amb"]}
CERT_COL = {"High": C["high"], "Medium": C["medium"], "Low": C["low"], "-": C["amb"]}


def _f(x, nd=1):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) or pd.isna(x) else round(float(x), nd)


def _arc(cx, cy, r, a0, a1):
    """SVG arc path from fraction a0 to a1 of the circle, clockwise from 12 o'clock."""
    if a1 - a0 >= 0.999:
        a1 = a0 + 0.999
    t0, t1 = 2 * math.pi * a0 - math.pi / 2, 2 * math.pi * a1 - math.pi / 2
    x0, y0, x1, y1 = cx + r * math.cos(t0), cy + r * math.sin(t0), cx + r * math.cos(t1), cy + r * math.sin(t1)
    large = 1 if (a1 - a0) > 0.5 else 0
    return f"M{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f}"


def _clock(cx, cy, pd_, pf, pu, direction, cert):
    r_ring, r_in = 19, 13
    out = [f'<circle cx="{cx}" cy="{cy}" r="{r_ring}" fill="none" stroke="{C["grid"]}" stroke-width="5"/>']
    a = 0.0
    for p, col in ((pd_, C["down"]), (pf, C["none"]), (pu, C["up"])):
        if p and p > 0.005:
            out.append(f'<path d="{_arc(cx, cy, r_ring, a, a + p)}" fill="none" stroke="{col}" stroke-width="5" '
                       f'stroke-linecap="butt"/>')
            a += p
    col = DIR_COL.get(direction, C["amb"])
    strong = cert in ("High", "Medium") and direction in ("Down", "Up", "None")
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r_in}" fill="{col if strong else C["bg"]}" '
               f'stroke="{col}" stroke-width="{0 if strong else 1.5}"/>')
    g = "#ffffff" if strong else col
    if direction == "Down":
        out.append(f'<path d="M{cx},{cy-7} L{cx},{cy+6} M{cx-5},{cy+1} L{cx},{cy+7} L{cx+5},{cy+1}" '
                   f'stroke="{g}" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    elif direction == "Up":
        out.append(f'<path d="M{cx},{cy+7} L{cx},{cy-6} M{cx-5},{cy-1} L{cx},{cy-7} L{cx+5},{cy-1}" '
                   f'stroke="{g}" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    elif direction == "None":
        out.append(f'<path d="M{cx-6},{cy} L{cx+6},{cy}" stroke="{g}" stroke-width="2.6" stroke-linecap="round"/>')
    else:   # no clear direction
        out.append(f'<path d="M{cx-6},{cy+1} q3,-5 6,0 t6,0" stroke="{g}" stroke-width="2" fill="none" '
                   f'stroke-linecap="round"/>')
    return "".join(out)


def build_html(f: pd.DataFrame, view: str = "Price", now_utc=None, scroll_to_now: bool = True) -> tuple[str, int]:
    """Returns (html, height). `f` columns: quarter_utc, t, lo/mid/hi/point/actual/ref (in the chosen
    view), spot_eur, q10/q50/q90/exp_spread/spread_actual (spread), p_down/p_flat/p_up, direction,
    certainty, action, mwh, decision, status, pnl_eur."""
    n = len(f)
    colw, left, top, ch = 56, 4, 26, 330
    lw = 74                                                     # fixed label column (does not scroll)
    w = left + n * colw + 8
    vals = pd.concat([f["lo"], f["hi"], f["point"], f["actual"], f["ref"]]).astype(float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vmin, vmax = -50.0, 50.0
    pad = max(10.0, 0.06 * (vmax - vmin))
    vmin, vmax = vmin - pad, vmax + pad
    step = [5, 10, 20, 25, 50, 100, 200, 250, 500][min(range(9), key=lambda i: abs((vmax - vmin) / [5, 10, 20, 25, 50, 100, 200, 250, 500][i] - 7))]
    y = lambda v: top + ch * (vmax - v) / (vmax - vmin)          # noqa: E731

    y_clock = top + ch + 46
    rows = {"Direction": y_clock + 44, "Certainty": y_clock + 70, "Decision": y_clock + 96, "Settled P&L": y_clock + 122}
    h = rows["Settled P&L"] + 22
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="display:block">']
    lab = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw}" height="{h}" style="display:block">']
    # grid + y labels
    g0 = math.ceil(vmin / step) * step
    v = g0
    while v <= vmax:
        yy = y(v)
        s.append(f'<line x1="{left}" x2="{w-8}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{C["grid"]}" stroke-width="1"/>')
        lab.append(f'<text x="{lw-6}" y="{yy+4:.1f}" fill="{C["axis"]}" font-size="11" text-anchor="end">{v:g}</text>')
        v += step
    if vmin < 0 < vmax:
        s.append(f'<line x1="{left}" x2="{w-8}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="{C["axis"]}" stroke-width="1.2"/>')
    lab.append(f'<text x="14" y="{top+ch/2}" fill="{C["axis"]}" font-size="12" transform="rotate(-90 14 {top+ch/2})" '
             f'text-anchor="middle">{view} (EUR/MWh)</text>')
    for lbl, yy in [("Time", y_clock - 30)] + [(k.replace("&", "&amp;"), v) for k, v in rows.items()]:
        lab.append(f'<text x="{lw-6}" y="{yy+4}" fill="{C["axis"]}" font-size="11" text-anchor="end">{lbl}</text>')

    now_utc = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    now_x = None
    data = []
    for i, r in enumerate(f.itertuples(index=False)):
        x0 = left + i * colw
        cx = x0 + colw / 2
        q = pd.Timestamp(r.quarter_utc)
        past = q + pd.Timedelta(minutes=15) <= now_utc
        if q <= now_utc < q + pd.Timedelta(minutes=15):
            now_x = x0
        if past:
            s.append(f'<rect x="{x0}" y="{top}" width="{colw}" height="{rows["Settled P&L"]+12-top}" fill="{C["past"]}"/>')
        s.append(f'<g class="q" data-i="{i}"><rect x="{x0}" y="{top}" width="{colw}" '
                 f'height="{rows["Settled P&L"]+12-top}" fill="transparent"/>')
        lo, hi, mid, pt, act, ref = (_f(getattr(r, k)) for k in ("lo", "hi", "mid", "point", "actual", "ref"))
        if view == "Price" and _f(r.spot_eur) is None:
            s.append(f'<text x="{cx}" y="{top+ch/2}" fill="#7f889b" font-size="11" text-anchor="middle" '
                     f'transform="rotate(70 {cx} {top+ch/2})">Spot price not yet set</text>')
        elif lo is not None and hi is not None:
            s.append(f'<rect x="{x0+9}" y="{y(hi):.1f}" width="{colw-18}" height="{max(2, y(lo)-y(hi)):.1f}" rx="3" '
                     f'fill="{C["band"]}" fill-opacity="0.88" stroke="{C["band_edge"]}" stroke-width="1"/>')
            if mid is not None:
                s.append(f'<line x1="{x0+9}" x2="{x0+colw-9}" y1="{y(mid):.1f}" y2="{y(mid):.1f}" '
                         f'stroke="{C["median"]}" stroke-width="2"/>')
            if ref is not None and view == "Price":
                s.append(f'<line x1="{x0+5}" x2="{x0+colw-5}" y1="{y(ref):.1f}" y2="{y(ref):.1f}" '
                         f'stroke="{C["spot"]}" stroke-width="2" stroke-dasharray="4,2"/>')
            if pt is not None:
                s.append(f'<circle cx="{cx}" cy="{y(pt):.1f}" r="4.5" fill="{C["point"]}" stroke="#ffffff" stroke-width="1.2"/>')
        if act is not None:
            ya = y(act)
            s.append(f'<path d="M{cx},{ya-7:.1f} L{cx+7},{ya:.1f} L{cx},{ya+7:.1f} L{cx-7},{ya:.1f} Z" '
                     f'fill="{C["actual"]}" stroke="#0b0d14" stroke-width="1.2"/>')
        s.append(f'<text x="{cx}" y="{y_clock-26}" fill="{C["text"]}" font-size="12" font-weight="600" '
                 f'text-anchor="middle">{r.t}</text>')
        s.append(_clock(cx, y_clock, _f(r.p_down, 3) or 0, _f(r.p_flat, 3) or 0, _f(r.p_up, 3) or 0,
                        r.direction, r.certainty))
        s.append(f'<text x="{cx}" y="{rows["Direction"]+4}" fill="{DIR_COL.get(r.direction, C["amb"])}" '
                 f'font-size="12" font-weight="700" text-anchor="middle">{r.direction}</text>')
        s.append(f'<text x="{cx}" y="{rows["Certainty"]+4}" fill="{CERT_COL.get(r.certainty, C["amb"])}" '
                 f'font-size="12" font-weight="700" text-anchor="middle">{r.certainty}</text>')
        if r.action in ("BUY", "SELL"):
            col = C["up"] if r.action == "BUY" else C["down"]
            s.append(f'<rect x="{x0+5}" y="{rows["Decision"]-9}" width="{colw-10}" height="18" rx="9" fill="{col}"/>'
                     f'<text x="{cx}" y="{rows["Decision"]+4}" fill="#0b0d14" font-size="11" font-weight="700" '
                     f'text-anchor="middle">{r.action} {_f(r.mwh)}</text>')
        else:
            s.append(f'<text x="{cx}" y="{rows["Decision"]+4}" fill="#8a93a6" font-size="11" text-anchor="middle">HOLD</text>')
        pnl = _f(r.pnl_eur, 0)
        if pnl is not None and r.action in ("BUY", "SELL") and _f(r.spread_actual) is not None:
            s.append(f'<text x="{cx}" y="{rows["Settled P&L"]+4}" fill="{C["high"] if pnl >= 0 else C["low"]}" '
                     f'font-size="11" font-weight="700" text-anchor="middle">{"+" if pnl >= 0 else ""}{pnl:g}</text>')
        s.append("</g>")
        data.append({"t": r.t, "status": r.status, "dir": r.direction, "cert": r.certainty, "dec": r.decision,
                     "pd": _f(r.p_down, 3), "pf": _f(r.p_flat, 3), "pu": _f(r.p_up, 3),
                     "hi": hi, "mid": mid, "lo": lo, "pt": pt, "spot": _f(r.spot_eur), "act": act,
                     "pnl": pnl if r.action in ("BUY", "SELL") else None})
    if now_x is not None:
        s.append(f'<line x1="{now_x}" x2="{now_x}" y1="{top-6}" y2="{rows["Settled P&L"]+12}" stroke="{C["now"]}" '
                 f'stroke-width="1.5" stroke-dasharray="3,3"/><text x="{now_x+3}" y="{top-10}" fill="{C["now"]}" '
                 f'font-size="11">now</text>')
    s.append("</svg>")
    lab.append("</svg>")
    legend = (f'<span style="color:{C["band"]}">■</span> 10-90% range &nbsp; <span style="color:#fff">━</span> median '
              f'&nbsp; <span style="color:{C["point"]}">●</span> expected &nbsp; '
              + (f'<span style="color:{C["spot"]}">┅</span> day-ahead (spot) &nbsp; ' if view == "Price" else "")
              + f'<span style="color:{C["actual"]}">◆</span> settled actual &nbsp; ring: '
              f'<span style="color:{C["down"]}">down</span> / <span style="color:{C["none"]}">none</span> / '
              f'<span style="color:{C["up"]}">up</span> probability')
    unit = "EUR/MWh"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:{C['bg']};font-family:Inter,Segoe UI,Arial,sans-serif;color:{C['text']}}}
.wrap{{background:{C['bg']};border-radius:8px;padding:8px 6px 4px}}
.leg{{font-size:12px;color:{C['axis']};padding:0 10px 6px}}
.sc{{overflow-x:auto;overflow-y:hidden}}
.q:hover>rect:first-child{{fill:rgba(255,255,255,0.06)}}
#tip{{position:fixed;display:none;pointer-events:none;background:#ffffff;color:#14161f;border-radius:6px;
 box-shadow:0 4px 16px rgba(0,0,0,.45);padding:10px 12px;font-size:12px;min-width:210px;z-index:9}}
#tip h4{{margin:0 0 6px;font-size:13px}} #tip td{{padding:1px 6px 1px 0}} .bar{{height:10px;border-radius:2px}}
</style></head><body><div class="wrap"><div class="leg">{legend}</div>
<div style="display:flex"><div style="flex:0 0 {lw}px">{''.join(lab)}</div><div class="sc" id="sc" style="flex:1">{''.join(s)}</div></div></div><div id="tip"></div>
<script>
const D={json.dumps(data)}; const tip=document.getElementById('tip');
const f=(v)=>v===null?'-':'€'+v.toFixed(1); const p=(v)=>v===null?'-':Math.round(v*100)+'%';
function bar(lbl,v,c){{return `<tr><td>${{lbl}}</td><td style="width:110px"><div class="bar" style="width:${{Math.round((v||0)*100)}}%;background:${{c}}"></div></td><td>${{p(v)}}</td></tr>`}}
document.querySelectorAll('.q').forEach(g=>{{
 g.addEventListener('mousemove',e=>{{const d=D[+g.dataset.i];
  tip.innerHTML=`<h4>${{d.t}} &middot; ${{d.status}}</h4>
  <b>Most likely direction: <span style="color:${{({json.dumps(DIR_COL)})[d.dir]}}">${{d.dir}}</span></b><br>Certainty: <b>${{d.cert}}</b>
  <table style="margin:6px 0">${{bar('Down',d.pd,'{C['down']}')}}${{bar('None',d.pf,'{C['none']}')}}${{bar('Up',d.pu,'{C['up']}')}}</table>
  <table><tr><td>90%</td><td>${{f(d.hi)}}</td></tr><tr><td>Expected (point)</td><td style="color:{C['point']}"><b>${{f(d.pt)}}</b></td></tr>
  <tr><td>50% (median)</td><td>${{f(d.mid)}}</td></tr><tr><td>10%</td><td>${{f(d.lo)}}</td></tr>
  <tr><td>Day-ahead (spot)</td><td>${{f(d.spot)}}</td></tr><tr><td>Settled actual</td><td><b>${{f(d.act)}}</b></td></tr></table>
  <div style="margin-top:6px">Decision: <b>${{d.dec}}</b>${{d.pnl===null?'':' &middot; P&amp;L <b>'+(d.pnl>=0?'+':'')+'€'+d.pnl+'</b>'}}</div>
  <div style="color:#666;margin-top:4px;font-size:11px">{view} view, {unit}</div>`;
  tip.style.display='block'; let x=e.clientX+14, yv=e.clientY+10;
  if(x+tip.offsetWidth>window.innerWidth) x=e.clientX-tip.offsetWidth-14;
  if(yv+tip.offsetHeight>window.innerHeight) yv=window.innerHeight-tip.offsetHeight-4;
  tip.style.left=x+'px'; tip.style.top=yv+'px';}});
 g.addEventListener('mouseleave',()=>tip.style.display='none');}});
{f"const nx={now_x}; if(nx) document.getElementById('sc').scrollLeft=Math.max(0,nx-{left}-{colw*2});" if (now_x is not None and scroll_to_now) else ""}
</script></body></html>"""
    return html, h + 50
