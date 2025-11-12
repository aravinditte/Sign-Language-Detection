"""
Sign Language Detection - Kaggle Dataset Downloader (with .env support)
Author: Aravind
Description: Downloads Sign Language MNIST dataset from Kaggle using credentials from .env
"""

import os
import zipfile
from dotenv import load_dotenv
import shutil

# Load .env
load_dotenv()
kaggle_json_path = os.getenv("KAGGLE_JSON", "./kaggle.json")

# Move kaggle.json to ~/.kaggle/
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

# Download dataset
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("[ERROR] kaggle package is not installed.\nRun: pip install kaggle python-dotenv")
    exit(1)

api = KaggleApi()
api.authenticate()

data_dir = 'data/kaggle'
os.makedirs(data_dir, exist_ok=True)

print("\nDownloading Sign Language MNIST dataset from Kaggle ...\n")
try:
    api.dataset_download_files(
        'datamunge/sign-language-mnist',
        path=data_dir,
        unzip=True
    )
    print(f"\n[INFO] Dataset downloaded & extracted to {data_dir}\n")
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            size = os.path.getsize(os.path.join(data_dir, file)) / (1024*1024)
            print(f"  - {file} ({size:.2f} MB)")
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    print("\nManual download instructions:")
    print("1. Go to: https://www.kaggle.com/datasets/datamunge/sign-language-mnist\n2. Download manually and place CSVs in data/kaggle/")
    exit(1)
