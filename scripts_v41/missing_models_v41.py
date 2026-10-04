"""Print 'AREA:family' for every zone whose RECOMMENDED model file is not on this machine yet
(config_v41.yaml models.recommended). Used by pull_on_denmark_v41.bat step 7. 2026-10-04."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Nurex_V4_2"))
from v4_1_intraday import settings as S, pipeline_id as P  # noqa: E402

c = S.load(mode="batch")
print(" ".join(f"{a}:{P.recommended_family(c, a)}" for a in c["areas"] if not P.model_path(c, a).exists()))
