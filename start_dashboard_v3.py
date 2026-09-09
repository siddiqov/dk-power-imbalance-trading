# ==============================================================================
# start_dashboard_v3.py
# Production WSGI / Flask Runner for V3 Optimeering Commercial Dashboard
# ==============================================================================

import os
import sys
import argparse

from dashboard_v3 import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V3 Optimeering Energy Trading Simulator')
    parser.add_argument('--port', type=int, default=5001, help='Port to run Flask dashboard on (default: 5001)')
    parser.add_argument('--volume', type=float, default=2.0, help='Default trade volume in MWh')
    parser.add_argument('--area', type=str, default='DK1', choices=['DK1', 'DK2'], help='Price Area (DK1/DK2)')
    args = parser.parse_args()

    port = int(os.environ.get('PORT', args.port))
    print('\n' + '=' * 80)
    print(f'  V3 OPTIMEERING COMMERCIAL SIMULATOR RUNNING ON http://127.0.0.1:{port}')
    print(f'  Price Area: {args.area} | Volume Sizing: {args.volume} MWh/quarter')
    print('=' * 80, flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
