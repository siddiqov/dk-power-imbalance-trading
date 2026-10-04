"""Quick check of the V4.1 models the dashboard and the cycle use (2026-10-04).

  python check_models_v41.py [YYYY-MM-DD]

Per zone: recommended model (locked + sent to the client) and comparison model, their thresholds,
and one local day of decisions for both at the Validated and Aggressive levels (risk limits on).
Read-only: nothing is trained, locked or written except the console output.
"""
import sys
import time

import pandas as pd

sys.path.insert(0, "Nurex_V4_2")
from v4_1_intraday import dashboard_adapter as A  # noqa: E402

day = sys.argv[1] if len(sys.argv) > 1 else (pd.Timestamp.now(tz="Europe/Copenhagen") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
print(f"pandas {pd.__version__} | day checked: {day}")
ok = True
for area in ("DK1", "DK2"):
    rec, cmp_ = A.recommended_family(area), A.compare_family(area)
    print(f"\n=== {area}: recommended (client) = {A.family_label(rec)} | comparison = {A.family_label(cmp_) if cmp_ else '-'}")
    for fam in [f for f in (rec, cmp_) if f]:
        b = A.load_bundle(area, fam)
        if b is None:
            print(f"  {fam}: MODEL FILE MISSING")
            ok = False
            continue
        lr = {k: v for k, v in (b.decision_params.get("logreg") or {}).items() if k != "candidates"}
        print(f"  {fam}: {type(b.model).__name__} {b.version} trained_until={b.trained_until} "
              f"buy={b.decision_params.get('buy')} sell={b.decision_params.get('sell')} {lr or ''}")
        t0 = time.time()
        try:
            res = A.day_decisions_risked(area, day, levels=("validated", "aggressive"), family=fam)
        except Exception as e:
            print(f"  {fam}: day_decisions FAILED: {type(e).__name__}: {e}")
            ok = False
            continue
        for lvl, d in res.items():
            tr = d[d["action"] != "HOLD"]
            print(f"    {lvl:<10} trades={len(tr):3d} buy={int((tr['action'] == 'BUY').sum()):3d} "
                  f"sell={int((tr['action'] == 'SELL').sum()):3d} MWh={tr['mwh'].sum():6.1f} "
                  f"settled net={d.loc[d['settled'], 'pnl_eur'].sum():8.2f} EUR "
                  f"locked rows={int((d.get('source', pd.Series(dtype=str)) == 'LOCKED').sum())}")
        print(f"    ({time.time() - t0:.0f}s)")
print("\nRESULT:", "OK" if ok else "PROBLEMS - see above")
