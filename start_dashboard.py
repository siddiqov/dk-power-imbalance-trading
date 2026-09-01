# ==============================================================================
# start_dashboard.py
# Production WSGI / Flask Runner for V2 Commercial Dashboard
# ==============================================================================

import os
import sys

from dashboard_v2 import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 80)
    print(f"  V2 COMMERCIAL TRADING SIMULATOR RUNNING ON http://127.0.0.1:{port}")
    print("=" * 80, flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
