# ==============================================================================
# start_dashboard.py
# Production WSGI / Flask Runner for V2 Commercial Dashboard
# Supports multi-port and multi-volume execution (e.g. 2.0 MWh, 5.0 MWh, 10.0 MWh)
# ==============================================================================

import os
import sys
import argparse

from dashboard_v2 import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="V2 Commercial Energy Trading Simulator")
    parser.add_argument("--port", type=int, default=5000, help="Port to run Flask dashboard on (default: 5000)")
    parser.add_argument("--volume", type=float, default=2.0, help="Default trade volume in MWh (e.g., 2.0, 5.0, 10.0)")
    parser.add_argument("--area", type=str, default="DK1", choices=["DK1", "DK2"], help="Price Area (DK1/DK2)")
    args = parser.parse_args()

    port = int(os.environ.get('PORT', args.port))
    print("\n" + "=" * 80)
    print(f"  V2 COMMERCIAL TRADING SIMULATOR RUNNING ON http://127.0.0.1:{port}")
    print(f"  Price Area: {args.area} | Volume Sizing: {args.volume} MWh/quarter")
    print("=" * 80, flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
