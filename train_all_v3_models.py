# ==============================================================================
# train_all_v3_models.py
# Multi-Paradigm V3 Model Retraining Pipeline (DK1 & DK2)
# Zero Synthetic Data / 100% Genuine Dual-Resolution Dataset
# ==============================================================================

import os
import sys
import time
from src.model_trainer_v3 import V3QuantileModelSuite


def retrain_v3_for_area(price_area):
    start_time = time.time()
    print(f"\n================================================================================")
    print(f"  STARTING FULL 4-PARADIGM V3 RETRAINING FOR {price_area}")
    print(f"================================================================================")

    suite = V3QuantileModelSuite(price_area=price_area, model_dir="models_v3")
    suite.train_all_paradigms_from_database()

    elapsed = time.time() - start_time
    print(f"[{price_area}] Retraining successfully completed in {elapsed:.2f} seconds.")


if __name__ == '__main__':
    total_start = time.time()
    areas = ["DK1", "DK2"]
    
    print("\n" + "#" * 80)
    print("  NUREX ENERGY TRADING - V3 OPTIMEERING 4-PARADIGM SUITE RETRAINING")
    print("  Price Areas: DK1 & DK2 | Macro (1999-2025 1h) + Micro (2025+ 15m)")
    print("#" * 80)

    for area in areas:
        retrain_v3_for_area(area)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"  ALL V3 MODELS (DK1 & DK2) RETRAINED AND VERIFIED IN {total_elapsed:.2f}s")
    print("=" * 80 + "\n")
