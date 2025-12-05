# API Documentation

## Data Loader Module

### `GoogleDriveLoader`
Load reservoir data from Google Drive.

```python
from src.data_loader import GoogleDriveLoader

# Initialize loader
loader = GoogleDriveLoader(credentials_path='credentials.json')

# Load data from Google Drive links
drive_links = ["https://drive.google.com/file/d/FILE_ID/view"]
data = loader.load_from_drive(drive_links)
