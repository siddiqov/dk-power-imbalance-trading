# ==============================================================================
# start_dashboard_v4.py
# Dedicated Launcher for Nurex V4.0 Institutional Dashboard on Port 5005
# ==============================================================================

import os
import sys
import subprocess

def start_dashboard():
    port = 5005
    print(f"Starting Nurex V4.0 Dashboard on http://127.0.0.1:{port}/ ...")
    
    streamlit_exe = r"C:\Users\Hafeez\anaconda3\Scripts\streamlit.exe"
    if not os.path.exists(streamlit_exe):
        streamlit_exe = "streamlit"

    cmd = [
        streamlit_exe,
        "run",
        "dashboard_v4.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true"
    ]
    
    proc = subprocess.Popen(cmd)
    print(f"V4.0 Dashboard successfully launched in background (PID: {proc.pid}) on port {port}.")

if __name__ == "__main__":
    start_dashboard()
