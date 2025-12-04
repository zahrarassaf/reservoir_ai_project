# این کد جدید را اجرا کنید
# scripts/download_and_analyze.py

import gdown
import os
from pathlib import Path

def download_spe9_from_drive():
    """Download SPE9 data from your Google Drive link."""
    
    # لینک مستقیم دانلود (ID فایل)
    file_id = "1Ue_EHX8w2h8WlT9kGdL3jFjF1b3yLnfL"
    
    # آدرس خروجی
    output_path = "data/spe9_data.tar.gz"
    
    # ایجاد دایرکتوری
    Path("data").mkdir(exist_ok=True)
    
    print(f"📥 Downloading SPE9 data from Google Drive...")
    print(f"   File ID: {file_id}")
    print(f"   Output: {output_path}")
    
    try:
        # دانلود با gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        
        print(f"✅ Download complete!")
        return output_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n🔧 Try installing gdown first:")
        print("   pip install gdown")
        return None

# اجرای دانلود
if __name__ == "__main__":
    downloaded_file = download_spe9_from_drive()
    
    if downloaded_file:
        print(f"\n🎯 Now run: python scripts/analyze_real_spe9.py")
        print(f"   And enter this path: {downloaded_file}")
