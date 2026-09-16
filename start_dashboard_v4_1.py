# ==============================================================================
# start_dashboard_v4_1.py
# Launcher for Nurex V4.1 Institutional High-Alpha Dashboard on Port 5005
# ==============================================================================

import subprocess
import sys
import os

if __name__ == "__main__":
    print("=" * 65)
    print("  LAUNCHING NUREX V4.1 INSTITUTIONAL DASHBOARD ON PORT 5005")
    print("  URL: http://127.0.0.1:5005/")
    print("=" * 65)
    
    cmd = [
        r"C:\Users\Hafeez\anaconda3\Scripts\streamlit.exe",
        "run",
        "dashboard_v4_1.py",
        "--server.port", "5005",
        "--server.headless", "true"
    ]
    
    subprocess.run(cmd)
