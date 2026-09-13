#!/usr/bin/env python3
# ==============================================================================
# scripts/manage_service.py
# Management CLI for Nurex 24/7 Power Trading Background Service
#
# Commands:
#   python scripts/manage_service.py status  -> Check task and process health
#   python scripts/manage_service.py start   -> Start scheduled task service
#   python scripts/manage_service.py stop    -> Stop scheduled task and kill process
#   python scripts/manage_service.py restart -> Restart service
#   python scripts/manage_service.py logs    -> View recent 40 log lines
# ==============================================================================

import sys
import os
import time
import subprocess
import psutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

TASK_NAME = "Nurex_PowerTrading_AutoPublisher"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "auto_publisher.log")


def get_publisher_processes():
    procs = []
    current_pid = os.getpid()
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'status']):
        try:
            if p.pid == current_pid:
                continue
            cmd = " ".join(p.info['cmdline'] or [])
            if "auto_publisher" in cmd:
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs


def get_task_status():
    try:
        res = subprocess.run(
            ["powershell", "-Command", f"(Get-ScheduledTask -TaskName '{TASK_NAME}' -ErrorAction SilentlyContinue).State"],
            capture_output=True,
            text=True
        )
        state = res.stdout.strip()
        return state if state else "Not Registered"
    except Exception as e:
        return f"Error: {e}"


def cmd_status():
    print("=" * 70)
    print("📊 NUREX 24/7 POWER TRADING SERVICE STATUS AUDIT")
    print("=" * 70)
    task_state = get_task_status()
    print(f"  Windows Task Scheduler: '{TASK_NAME}'")
    print(f"  Scheduler Task State:   {task_state}")
    
    procs = get_publisher_processes()
    if procs:
        print(f"\n  Active Background Processes: ({len(procs)} found)")
        for p in procs:
            print(f"    - PID: {p.pid} | Status: {p.status()} | CMD: {' '.join(p.cmdline()[:4])}")
    else:
        print("\n  Active Background Processes: None currently running.")
        
    print(f"\n  Log File: {LOG_FILE}")
    if os.path.exists(LOG_FILE):
        size_kb = os.path.getsize(LOG_FILE) / 1024.0
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(LOG_FILE)))
        print(f"  Log File Size: {size_kb:.1f} KB (Last modified: {mtime})")
    else:
        print("  Log File: Not yet created.")
    print("=" * 70)


def cmd_stop():
    print(f"Stopping Windows Scheduled Task '{TASK_NAME}'...")
    subprocess.run(["powershell", "-Command", f"Stop-ScheduledTask -TaskName '{TASK_NAME}' -ErrorAction SilentlyContinue"], capture_output=True)
    
    procs = get_publisher_processes()
    count = 0
    for p in procs:
        try:
            p.kill()
            count += 1
            print(f"  Killed process PID {p.pid}")
        except Exception:
            pass
    time.sleep(1)
    print(f"✅ Service stopped ({count} processes terminated).")


def cmd_start():
    cmd_stop()
    print(f"Starting Windows Scheduled Task '{TASK_NAME}'...")
    subprocess.run(["powershell", "-Command", f"Start-ScheduledTask -TaskName '{TASK_NAME}'"], capture_output=True)
    time.sleep(2)
    cmd_status()


def cmd_restart():
    print("Restarting service...")
    cmd_start()


def cmd_logs(lines=40):
    if not os.path.exists(LOG_FILE):
        print(f"Log file '{LOG_FILE}' does not exist.")
        return
    print(f"--- [Last {lines} lines of {LOG_FILE}] ---")
    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line, end="")
    print("\n-------------------------------------------")


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "status":
        cmd_status()
    elif action == "start":
        cmd_start()
    elif action == "stop":
        cmd_stop()
    elif action == "restart":
        cmd_restart()
    elif action == "logs":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        cmd_logs(lines)
    else:
        print(f"Unknown action: {action}. Valid options: status | start | stop | restart | logs")


if __name__ == "__main__":
    main()
