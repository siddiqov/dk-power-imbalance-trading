import os
import sys
import subprocess

def start_dashboard():
    port = 5003
    print(f"Starting Nurex V3.2 Dashboard on http://127.0.0.1:{port}/ ...")
    streamlit_exe = r"C:\Users\Hafeez\anaconda3\Scripts\streamlit.exe"
    if not os.path.exists(streamlit_exe):
        streamlit_exe = "streamlit"
    cmd = [streamlit_exe, "run", "dashboard_v3_2.py", "--server.port", str(port), "--server.headless", "true"]
    proc = subprocess.Popen(cmd)
    print(f"V3.2 Dashboard successfully launched in background (PID: {proc.pid}) on port {port}.")

if __name__ == "__main__":
    start_dashboard()

