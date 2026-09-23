"""Shim: run.py launches this via streamlit; the real dashboard lives at ../dashboard_v4_1.py"""
import sys, pathlib, importlib, time

# Force-evict stale cached modules so hot-reload picks up code changes
_stale = [k for k in list(sys.modules.keys())
          if any(x in k for x in ['model_trainer_v3_1', 'commercial_strategy_v3_1',
                                   'feature_engineering_v3', 'hyperparameter_optimizer_v3_1'])]
for _mod_name in _stale:
    del sys.modules[_mod_name]

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
exec(compile(
    open(pathlib.Path(__file__).resolve().parents[2] / "dashboard_v4_1.py", encoding="utf-8").read(),
    "dashboard_v4_1.py", "exec"))
# RELOAD: 1790106690.1793683
