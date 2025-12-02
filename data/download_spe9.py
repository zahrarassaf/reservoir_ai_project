# data/download_spe9.py
import requests
import tarfile
import os
from pathlib import Path

class SPE9DataDownloader:
    """دانلود خودکار داده‌های استاندارد صنعت نفت"""
    
    SPE9_URL = "https://github.com/OPM/opm-data/raw/master/spe9/SPE9_CP.DATA"
    
    def __init__(self, data_dir="data/spe9"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self):
        """دانلود فایل‌های داده SPE9"""
        print("📥 Downloading SPE9 benchmark dataset...")
        
        # دانلود فایل اصلی
        data_file = self.data_dir / "SPE9_CP.DATA"
        if not data_file.exists():
            response = requests.get(self.SPE9_URL, stream=True)
            with open(data_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # دانلود فایل‌های اضافی اگر موجود باشند
        self._download_additional_files()
        
        print(f"✅ Data downloaded to {self.data_dir}")
        return self.data_dir
    
    def parse_eclipse_data(self, data_file):
        """پارس کردن فایل DATA اکلایپس"""
        # پارس کردن فایل ورودی اکلایپس
        # استخراج گرید، خواص سنگ، شرایط مرزی و...
        pass
