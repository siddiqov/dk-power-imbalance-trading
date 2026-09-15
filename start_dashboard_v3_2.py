import os
import sys
import subprocess

def start_dashboard():
    print("Starting V3.2 Dashboard on port 5004...")
    subprocess.Popen(["streamlit", "run", "dashboard_v3_2.py", "--server.port", "5004"])

if __name__ == "__main__":
    start_dashboard()
