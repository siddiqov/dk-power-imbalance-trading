# ==============================================================================
# start_dashboard_v4_1.py
# Launcher for Nurex V4.1 Institutional High-Alpha Dashboard on Port 5005
#
# Runs the dashboard with the SAME interpreter that trains the models:
# Nurex_V4_2/.venv (Python 3.14, numpy 2.5.3, scikit-learn 1.9.1, duckdb 1.5.5).
# Anaconda's Python carries numpy 1.x, which cannot unpickle a numpy 2 model file
# ("No module named 'numpy._core.numeric'"), so the V4.1 bundle silently fails to
# load there and every quarter falls back to "HOLD (model not trained)".
# ==============================================================================

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_STREAMLIT = os.path.join(HERE, "Nurex_V4_2", ".venv", "Scripts", "streamlit.exe")
FALLBACK_STREAMLIT = r"C:\Users\Hafeez\anaconda3\Scripts\streamlit.exe"

if __name__ == "__main__":
    print("=" * 65)
    print("  LAUNCHING NUREX V4.1 INSTITUTIONAL DASHBOARD ON PORT 5005")
    print("  URL: http://127.0.0.1:5005/")

    exe = VENV_STREAMLIT
    if os.path.exists(exe):
        print("  Interpreter: Nurex_V4_2/.venv  (matches the training environment)")
    else:
        exe = FALLBACK_STREAMLIT
        print("  WARNING: Nurex_V4_2/.venv not found - falling back to Anaconda.")
        print("           The V4.1 model may fail to load there (numpy 1.x vs 2.x).")
    print("=" * 65)

    if not os.path.exists(exe):
        sys.exit(f"streamlit not found at {exe}")

    subprocess.run([exe, "run", "dashboard_v4_1.py",
                    "--server.port", "5005", "--server.headless", "true"], cwd=HERE)
