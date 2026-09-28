"""Make a consistent private copy of the live V4.2 store for long read-only jobs (batch setup),
so they never block the 15-minute live cycle. The copy is taken only while a read-only
connection is held (= no writer active), exactly like the dashboard snapshot."""
import shutil, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from v4_1_intraday import settings as S          # noqa: E402
from nurex42.storage import Store                  # noqa: E402

dst = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "Nurex_V4_2" / "data" / "nurex42.setup_copy.duckdb"
cfg = S.load()
src = Path(cfg.db_path)
for i in range(40):                                # up to ~10 minutes
    try:
        st = Store(src, read_only=True)
    except Exception as e:
        print(f"store busy ({type(e).__name__}), retry {i + 1}/40 in 15 s", flush=True)
        time.sleep(15)
        continue
    try:
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
        wal, dwal = Path(str(src) + ".wal"), Path(str(dst) + ".wal")
        if wal.exists():
            shutil.copyfile(wal, dwal)
        elif dwal.exists():
            dwal.unlink()
    finally:
        st.close()
    print(f"copied {src.name} -> {dst}  ({dst.stat().st_size / 1e6:.0f} MB)")
    sys.exit(0)
print("could not get a quiet moment to copy the store")
sys.exit(1)
