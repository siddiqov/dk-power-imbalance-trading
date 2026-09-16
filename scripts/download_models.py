"""
Helper script to download pre-trained machine learning and deep learning models (V2, V3, V3.1, V3.2, V4, V4.1)
from Google Drive into the local 'models/', 'models_v3/', 'models_v3_1/', 'models_v3_2/', 'models_v4/', and 'models_v4_1/' directories.

Usage:
    python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_V3_DIR = BASE_DIR / "models_v3"
MODELS_V3_1_DIR = BASE_DIR / "models_v3_1"
MODELS_V3_2_DIR = BASE_DIR / "models_v3_2"
MODELS_V4_DIR = BASE_DIR / "models_v4"
MODELS_V4_1_DIR = BASE_DIR / "models_v4_1"

# --------------------------------------------------------------------------
# GOOGLE DRIVE CONFIGURATION
# Shared Folder ID for the complete pre-trained models repository
# --------------------------------------------------------------------------
GDRIVE_FOLDER_ID = "1xDMe236gDmZ4jUcXXgWoFsdITAxMteWU"

def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODELS_V3_DIR, exist_ok=True)
    os.makedirs(MODELS_V3_1_DIR, exist_ok=True)
    os.makedirs(MODELS_V3_2_DIR, exist_ok=True)
    os.makedirs(MODELS_V4_DIR, exist_ok=True)
    os.makedirs(MODELS_V4_1_DIR, exist_ok=True)
    print(f"[INFO] Target models directories:\n  - {MODELS_DIR}\n  - {MODELS_V3_DIR}\n  - {MODELS_V3_1_DIR}\n  - {MODELS_V3_2_DIR}\n  - {MODELS_V4_DIR}\n  - {MODELS_V4_1_DIR}")

    try:
        import gdown
    except ImportError:
        print("[ERROR] 'gdown' is required. Please install it using: pip install gdown")
        sys.exit(1)

    if GDRIVE_FOLDER_ID:
        print(f"[INFO] Downloading models folder from Google Drive (Folder ID: {GDRIVE_FOLDER_ID})...")
        url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}?usp=sharing"
        
        max_retries = 5
        import time
        for attempt in range(max_retries):
            try:
                gdown.download_folder(url=url, output=str(BASE_DIR / "gdrive_download"), quiet=False)
                print("[SUCCESS] All model assets downloaded from Google Drive successfully!")
                return
            except Exception as e:
                print(f"[WARNING] Download attempt {attempt + 1} failed due to: {e}. Retrying...")
                time.sleep(2)
        print("[NOTICE] Direct folder download hit rate limit. Individual file sync is available.")

if __name__ == "__main__":
    download_models()
