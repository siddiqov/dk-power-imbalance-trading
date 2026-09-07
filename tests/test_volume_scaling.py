import sys, os
sys.path.append(os.path.abspath('.'))
import unittest
from src.live_intraday_ledger import LiveIntradayLedger

class TestVolumeScaling(unittest.TestCase):

    def test_volume_scaling_dk1(self):
        # 1. Test 2.0 MWh
        ledger_2 = LiveIntradayLedger(price_area='DK1', capital=100000.0, trade_volume_mwh=2.0)
        res_2 = ledger_2.get_live_today_ledger('Transformer-TFT')

        # 2. Test 5.0 MWh
        ledger_5 = LiveIntradayLedger(price_area='DK1', capital=100000.0, trade_volume_mwh=5.0)
        res_5 = ledger_5.get_live_today_ledger('Transformer-TFT')

        # 3. Test 10.0 MWh
        ledger_10 = LiveIntradayLedger(price_area='DK1', capital=100000.0, trade_volume_mwh=10.0)
        res_10 = ledger_10.get_live_today_ledger('Transformer-TFT')

        print(f"DK1 2.0 MWh Fees: EUR {res_2['fees_so_far']:.2f} | Gross PnL: EUR {res_2['gross_pnl_so_far']:.2f}")
        print(f"DK1 5.0 MWh Fees: EUR {res_5['fees_so_far']:.2f} | Gross PnL: EUR {res_5['gross_pnl_so_far']:.2f}")
        print(f"DK1 10.0 MWh Fees: EUR {res_10['fees_so_far']:.2f} | Gross PnL: EUR {res_10['gross_pnl_so_far']:.2f}")

        # Assert fees scale proportionally
        self.assertAlmostEqual(res_5['fees_so_far'], res_2['fees_so_far'] * 2.5, places=2)
        self.assertAlmostEqual(res_10['fees_so_far'], res_2['fees_so_far'] * 5.0, places=2)

        # Assert gross PnL scales proportionally
        self.assertAlmostEqual(res_5['gross_pnl_so_far'], res_2['gross_pnl_so_far'] * 2.5, places=2)
        self.assertAlmostEqual(res_10['gross_pnl_so_far'], res_2['gross_pnl_so_far'] * 5.0, places=2)

        # Assert individual trade volume
        for t in res_10['trades']:
            if t['action'] != 'HOLD':
                self.assertEqual(t['volume_mwh'], 10.0)

        print("\n[SUCCESS] Volume scaling mathematical integrity verified 100%!")

if __name__ == '__main__':
    unittest.main()
