"""
ASL Alphabet Dataset Download Script
Author: Aravind
Description: Downloads and extracts ASL Alphabet dataset from Kaggle
"""
import os
from dotenv import load_dotenv
import shutil

# Load .env for Kaggle JSON credentials
load_dotenv()
kaggle_json_path = os.getenv("KAGGLE_JSON", "./kaggle.json")
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
dst = os.path.join(kaggle_dir, "kaggle.json")
if not os.path.exists(dst):
    if not os.path.exists(kaggle_json_path):
        print(f"[ERROR] kaggle.json not found. Set KAGGLE_JSON in your .env file or put it in the project root.")
        exit(1)
    shutil.copy2(kaggle_json_path, dst)
    os.chmod(dst, 0o600)
    print(f"[INFO] Copied kaggle.json to {dst}")
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("[ERROR] kaggle package is not installed.\nRun: pip install kaggle python-dotenv")
    exit(1)
api = KaggleApi()
api.authenticate()
data_dir = 'data/asl_alphabet'
os.makedirs(data_dir, exist_ok=True)
print("\nDownloading ASL Alphabet dataset from Kaggle ...\n")
try:
    api.dataset_download_files(
        'grassknoted/asl-alphabet',
        path=data_dir,
        unzip=True
    )
    print(f"\n[INFO] Dataset downloaded & extracted to {data_dir}\n")
    print("[INFO] Top-level extracted folders/files:")
    for folder in os.listdir(data_dir):
        print(f"  - {folder}")
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    print("\nManual download instructions:")
    print("1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet\n2. Download manually and extract to data/asl_alphabet/")
    exit(1)
