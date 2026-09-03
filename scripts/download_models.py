"""
Helper script to download pre-trained machine learning and deep learning models
from Google Drive into the local 'models/' directory.

Usage:
    python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# --------------------------------------------------------------------------
# GOOGLE DRIVE CONFIGURATION
# Replace these with your team's Google Drive shared Folder ID or File IDs
# --------------------------------------------------------------------------
GDRIVE_FOLDER_ID = "1xDMe236gDmZ4jUcXXgWoFsdITAxMteWU"

# Individual model files map: {filename: gdrive_file_id}
MODEL_FILES = {
    # "DK1_model_5future.pkl": "YOUR_GDRIVE_FILE_ID_HERE",
    # "DK1_model_multi_quarter.pkl": "YOUR_GDRIVE_FILE_ID_HERE",
    # "DK1_trained_bundle.pkl": "YOUR_GDRIVE_FILE_ID_HERE",
    # "DK1_gru_24.pt": "YOUR_GDRIVE_FILE_ID_HERE",
    # "DK1_gru_96.pt": "YOUR_GDRIVE_FILE_ID_HERE",
    # "DK1_lstm_24.pt": "YOUR_GDRIVE_FILE_ID_HERE",
    # "DK1_lstm_96.pt": "YOUR_GDRIVE_FILE_ID_HERE",
    # "DK1_prophet.pkl": "YOUR_GDRIVE_FILE_ID_HERE",
    # "DK1_seq_meta.pkl": "YOUR_GDRIVE_FILE_ID_HERE",
}

def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[INFO] Target models directory: {MODELS_DIR}")

    try:
        import gdown
    except ImportError:
        print("[ERROR] 'gdown' is required. Please install it using: pip install gdown")
        sys.exit(1)

    if GDRIVE_FOLDER_ID:
        print(f"[INFO] Downloading full models folder from Google Drive (ID: {GDRIVE_FOLDER_ID})...")
        url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}?usp=sharing"
        
        max_retries = 10
        import time
        for attempt in range(max_retries):
            try:
                gdown.download_folder(url=url, output=str(MODELS_DIR), quiet=False)
                print("[SUCCESS] All model assets downloaded successfully!")
                return
            except Exception as e:
                print(f"[WARNING] Download attempt {attempt + 1} failed due to: {e}. Retrying in 2 seconds...")
                time.sleep(2)
        print("[ERROR] Max retries reached. Download failed.")
        sys.exit(1)

    active_files = {k: v for k, v in MODEL_FILES.items() if not v.startswith("YOUR_")}
    if not active_files:
        print("\n" + "="*70)
        print("[NOTICE] Google Drive Folder ID or File IDs have not been set yet.")
        print("To configure automated model downloads:")
        print("1. Upload your 'models/' folder to Google Drive.")
        print("2. Set share permissions to 'Anyone with the link can view'.")
        print("3. Paste the Folder ID into GDRIVE_FOLDER_ID in 'scripts/download_models.py'.")
        print("="*70 + "\n")
        return

    for filename, file_id in active_files.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"[SKIP] '{filename}' already exists.")
            continue
        print(f"[DOWNLOADING] '{filename}'...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url=url, output=str(dest), quiet=False)

    print("[SUCCESS] Model download process finished.")

if __name__ == "__main__":
    download_models()
