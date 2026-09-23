Retired 2026-09-22: Nurex V4.2 dashboard (nurex42/dashboard.py, port 5006).
Port 5006 is now reserved for the V4.1-only dashboard.
RESTART_DASHBOARD.bat also force-killed EVERY python.exe on the PC (including the V4.1
dashboard and the V4.1 data cycle) and started "python run.py live" - do not reuse it.
"python run.py live" (V4.2 live cycle) itself is NOT retired and can still be started by hand.
