# -*- coding: utf-8 -*-
"""Nurex V4.2 pipeline module — implementation loaded from pre-compiled bytecode.
The original .py source is not available; the implementation lives in pipeline.pyc
(Python 3.14 bytecode). This loader makes the module importable on Python 3.14+.
"""
import importlib.util, pathlib as _p
_pyc = _p.Path(__file__).parent / "pipeline.pyc"
_spec = importlib.util.spec_from_file_location("nurex42.pipeline", _pyc)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("_")})
del _p, _pyc, _spec, _mod
