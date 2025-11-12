"""
Sign Language Detection - Kaggle Dataset Downloader
Author: Aravind
Description: Downloads Sign Language MNIST dataset from Kaggle
"""

import os
import zipfile
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def setup_kaggle_api():
    """
    Setup Kaggle API credentials
    
    Instructions:
    1. Go to https://www.kaggle.com/settings
    2. Click 'Create New API Token'
    3. Download kaggle.json
    4. Place it in ~/.kaggle/ (Linux/Mac) or C:\Users\YourName\.kaggle\ (Windows)
    5. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)
    """
    api = KaggleApi()
    api.authenticate()
    return api

def download_sign_language_mnist():
    """
    Download Sign Language MNIST dataset from Kaggle
    """
    print("="*60)
    print("Sign Language MNIST Dataset Downloader")
    print("="*60)
    
    # Create data directory
    data_dir = 'data/kaggle'
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Setup Kaggle API
        print("\nSetting up Kaggle API...")
        api = setup_kaggle_api()
        
        # Download dataset
        print("\nDownloading Sign Language MNIST dataset...")
        print("Dataset: datamunge/sign-language-mnist")
        
        api.dataset_download_files(
            'datamunge/sign-language-mnist',
            path=data_dir,
            unzip=True
        )
        
        print("\n" + "="*60)
        print("Dataset downloaded successfully!")
        print(f"Location: {data_dir}")
        print("\nFiles downloaded:")
        for file in os.listdir(data_dir):
            filepath = os.path.join(data_dir, file)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nPlease ensure:")
        print("1. You have a Kaggle account")
        print("2. You've accepted the dataset terms at:")
        print("   https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
        print("3. Your kaggle.json is in the correct location")
        print("4. Install kaggle: pip install kaggle")
        return False

def download_manually():
    """
    Instructions for manual download
    """
    print("\n" + "="*60)
    print("Manual Download Instructions")
    print("="*60)
    print("\nIf automatic download fails, follow these steps:")
    print("\n1. Go to: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
    print("2. Click 'Download' button")
    print("3. Extract the ZIP file")
    print("4. Place the CSV files in: data/kaggle/")
    print("   - sign_mnist_train.csv")
    print("   - sign_mnist_test.csv")
    print("\n5. Run the training script: python src/train_kaggle_model.py")
    print("="*60)

if __name__ == '__main__':
    success = download_sign_language_mnist()
    
    if not success:
        download_manually()
