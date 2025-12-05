"""
Google Drive Data Loader for Reservoir Simulation
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile

try:
    import gdown
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ReservoirData:
    """Reservoir data container"""
    time: np.ndarray
    production: pd.DataFrame
    pressure: np.ndarray
    injection: Optional[pd.DataFrame] = None
    petrophysical: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.petrophysical is None:
            self.petrophysical = pd.DataFrame()


class GoogleDriveLoader:
    """Load reservoir data from Google Drive"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Google Drive loader
        
        Parameters
        ----------
        credentials_path : str, optional
            Path to Google API credentials JSON file
        """
        self.credentials_path = credentials_path
        self.service = None
        
        if GOOGLE_AVAILABLE and credentials_path:
            self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API"""
        try:
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds = None
            
            token_file = Path('token.pickle')
            
            if token_file.exists():
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                with open(token_file, 'wb') as token:
                    pickle.dump(creds, token)
            
            self.service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive authentication successful")
            
        except Exception as e:
            logger.warning(f"Google Drive authentication failed: {e}")
            self.service = None
    
    def extract_file_id(self, url: str) -> str:
        """
        Extract file ID from Google Drive URL
        
        Parameters
        ----------
        url : str
            Google Drive URL
            
        Returns
        -------
        str
            File ID
        """
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/d/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return url
    
    def download_file(self, file_id: str, output_path: Path) -> bool:
        """
        Download file from Google Drive
        
        Parameters
        ----------
        file_id : str
            Google Drive file ID
        output_path : Path
            Output file path
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            # Try using gdown first (simpler)
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Downloaded {output_path.name} ({output_path.stat().st_size} bytes)")
                return True
            
        except Exception as e:
            logger.warning(f"gdown failed: {e}")
            
            # Fallback to Google API
            if self.service:
                try:
                    request = self.service.files().get_media(fileId=file_id)
                    fh = io.FileIO(output_path, 'wb')
                    downloader = MediaIoBaseDownload(fh, request)
                    
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        logger.info(f"Download {int(status.progress() * 100)}%")
                    
                    return True
                    
                except Exception as api_error:
                    logger.error(f"Google API download failed: {api_error}")
        
        return False
    
    def load_from_drive(self, drive_links: List[str]) -> ReservoirData:
        """
        Load reservoir data from Google Drive links
        
        Parameters
        ----------
        drive_links : List[str]
            List of Google Drive URLs
            
        Returns
        -------
        ReservoirData
            Structured reservoir data
        """
        logger.info(f"Loading data from {len(drive_links)} Google Drive links")
        
        # Create temporary directory for downloads
        temp_dir = Path(tempfile.mkdtemp())
        downloaded_files = []
        
        # Download all files
        for i, link in enumerate(drive_links):
            try:
                file_id = self.extract_file_id(link)
                output_path = temp_dir / f"reservoir_data_{i+1}.csv"
                
                logger.info(f"Downloading file {i+1}: {file_id}")
                if self.download_file(file_id, output_path):
                    downloaded_files.append(output_path)
                
            except Exception as e:
                logger.error(f"Error downloading {link}: {e}")
        
        # Process downloaded files
        data = self._process_files(downloaded_files)
        
        # Cleanup
        for file_path in downloaded_files:
            try:
                file_path.unlink()
            except:
                pass
        
        return data
    
    def _process_files(self, file_paths: List[Path]) -> ReservoirData:
        """
        Process downloaded CSV files
        
        Parameters
        ----------
        file_paths : List[Path]
            List of downloaded file paths
            
        Returns
        -------
        ReservoirData
            Processed reservoir data
        """
        all_data = {
            'time': None,
            'production': None,
            'pressure': None,
            'injection': None,
            'petrophysical': None
        }
        
        for file_path in file_paths:
            try:
                # Try to read the file
                df = self._read_data_file(file_path)
                
                # Auto-detect data type
                data_type = self._detect_data_type(df)
                
                if data_type == 'production':
                    all_data['production'] = self._process_production(df)
                elif data_type == 'pressure':
                    all_data['pressure'] = self._process_pressure(df)
                elif data_type == 'injection':
                    all_data['injection'] = self._process_injection(df)
                elif data_type == 'petrophysical':
                    all_data['petrophysical'] = self._process_petrophysical(df)
                elif data_type == 'time_series':
                    ts_data = self._process_time_series(df)
                    all_data.update(ts_data)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Create time array if missing
        if all_data['time'] is None:
            if all_data['production'] is not None:
                time_length = len(all_data['production'])
            elif all_data['pressure'] is not None:
                time_length = len(all_data['pressure'])
            else:
                time_length = 1826  # 5 years default
            
            all_data['time'] = np.arange(time_length)
        
        # Create ReservoirData object
        return ReservoirData(
            time=all_data['time'],
            production=all_data['production'] or pd.DataFrame(),
            pressure=all_data['pressure'] or np.array([]),
            injection=all_data['injection'],
            petrophysical=all_data['petrophysical'],
            metadata={
                'source': 'google_drive',
                'files_processed': len(file_paths)
            }
        )
    
    def _read_data_file(self, file_path: Path) -> pd.DataFrame:
        """Read data file with multiple format support"""
        try:
            # Try CSV first
            df = pd.read_csv(file_path)
        except:
            try:
                # Try Excel
                df = pd.read_excel(file_path)
            except:
                raise ValueError(f"Unsupported file format: {file_path}")
        
        return df
    
    def _detect_data_type(self, df: pd.DataFrame) -> str:
        """Detect type of reservoir data"""
        columns_lower = [str(col).lower() for col in df.columns]
        
        # Check for keywords
        if any('prod' in col or 'rate' in col or 'qo' in col for col in columns_lower):
            return 'production'
        elif any('press' in col or 'psi' in col for col in columns_lower):
            return 'pressure'
        elif any('inj' in col or 'wi' in col for col in columns_lower):
            return 'injection'
        elif any('poro' in col or 'perm' in col or 'phi' in col for col in columns_lower):
            return 'petrophysical'
        elif any('date' in col or 'time' in col or 'day' in col for col in columns_lower):
            return 'time_series'
        
        return 'unknown'
    
    def _process_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process production data"""
        # Find time column
        time_cols = [col for col in df.columns if 'date' in str(col).lower() or 'time' in str(col).lower()]
        
        if time_cols:
            df_processed = df.set_index(time_cols[0])
        else:
            df_processed = df
        
        # Convert to numeric
        df_processed = df_processed.apply(pd.to_numeric, errors='coerce')
        
        # Clean data
        df_processed = df_processed.fillna(method='ffill').fillna(0)
        df_processed = df_processed.clip(lower=0)
        
        return df_processed
    
    def _process_pressure(self, df: pd.DataFrame) -> np.ndarray:
        """Process pressure data"""
        # Find pressure column
        press_cols = [col for col in df.columns if 'press' in str(col).lower()]
        
        if press_cols:
            pressure = df[press_cols[0]].values
        else:
            # Assume first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            pressure = df[numeric_cols[0]].values if len(numeric_cols) > 0 else np.array([])
        
        # Clean pressure data
        pressure = pressure[~np.isnan(pressure)]
        pressure = pressure[pressure > 0]
        
        return pressure
    
    def _process_injection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process injection data"""
        return self._process_production(df)
    
    def _process_petrophysical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process petrophysical data"""
        # Standardize column names
        column_mapping = {
            'porosity': ['poro', 'phi', 'porosity'],
            'permeability': ['perm', 'k', 'permeability'],
            'netthickness': ['thick', 'h', 'thickness', 'net'],
            'watersaturation': ['sw', 'water_sat', 'saturation']
        }
        
        processed_df = pd.DataFrame()
        
        for std_name, variations in column_mapping.items():
            for col in df.columns:
                col_lower = str(col).lower()
                if any(var in col_lower for var in variations):
                    processed_df[std_name] = pd.to_numeric(df[col], errors='coerce')
                    break
        
        # Fill missing values
        processed_df = processed_df.fillna(processed_df.mean())
        
        return processed_df
    
    def _process_time_series(self, df: pd.DataFrame) -> Dict:
        """Process time series data"""
        result = {}
        
        # Extract time
        time_cols = [col for col in df.columns if 'date' in str(col).lower()]
        if time_cols:
            try:
                time_series = pd.to_datetime(df[time_cols[0]])
                result['time'] = (time_series - time_series.min()).dt.days.values
            except:
                pass
        
        # Extract production if present
        prod_cols = [col for col in df.columns if 'prod' in str(col).lower()]
        if prod_cols:
            result['production'] = df[prod_cols].apply(pd.to_numeric, errors='coerce')
        
        return result


def create_sample_data() -> ReservoirData:
    """
    Create sample reservoir data for testing
    
    Returns
    -------
    ReservoirData
        Sample reservoir data
    """
    np.random.seed(42)
    
    # Time data (5 years daily)
    time = np.arange(0, 5 * 365, 1)
    
    # Production data
    n_wells = 6
    production_data = {}
    
    for i in range(n_wells):
        base_rate = np.random.uniform(800, 3000)
        decline = np.random.uniform(0.0005, 0.005)
        
        # Exponential decline with noise
        production = base_rate * np.exp(-decline * time)
        noise = np.random.normal(0, base_rate * 0.05, len(time))
        production = np.maximum(production + noise, 0)
        
        production_data[f'Well_{i+1}'] = production
    
    # Pressure data
    initial_pressure = 4200
    pressure_decline = 0.4  # psi/day
    pressure = initial_pressure - pressure_decline * time / 365
    pressure += np.random.normal(0, 30, len(time))
    
    # Petrophysical data
    n_layers = 4
    petrophysical_data = {
        'porosity': np.random.uniform(0.12, 0.28, n_layers),
        'permeability': np.random.lognormal(3, 0.8, n_layers),
        'netthickness': np.random.uniform(15, 45, n_layers),
        'watersaturation': np.random.uniform(0.18, 0.35, n_layers)
    }
    
    return ReservoirData(
        time=time,
        production=pd.DataFrame(production_data, index=time),
        pressure=pressure,
        petrophysical=pd.DataFrame(petrophysical_data),
        metadata={
            'source': 'sample',
            'description': 'Synthetic reservoir data for demonstration'
        }
    )
