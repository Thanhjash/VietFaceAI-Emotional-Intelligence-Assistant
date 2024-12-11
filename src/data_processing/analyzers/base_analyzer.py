# D:\3.Project\VGU\CS_AGE\face_analysis\src\data_processing\base_analyzer.py

import abc
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class BaseAnalyzer(abc.ABC):
    """Base class for all analyzers providing common functionality"""
    
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize analyzer with data directory
        
        Args:
            data_dir: Path to AFAD dataset directory
            cache_dir: Optional path to cache computed results
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.metadata = None
        
    def scan_dataset(self) -> pd.DataFrame:
        """
        Scan dataset directory and create metadata DataFrame
        
        Returns:
            DataFrame with columns: [path, age, gender, filename]
        """
        data = []
        for age_dir in self.data_dir.glob("*"):
            if not age_dir.is_dir() or not age_dir.name.isdigit():
                continue
                
            age = int(age_dir.name)
            for gender_dir in age_dir.glob("11[12]"):
                gender = "male" if gender_dir.name == "111" else "female"
                
                for img_path in gender_dir.glob("*.jpg"):
                    data.append({
                        "path": str(img_path),
                        "age": age,
                        "gender": gender,
                        "filename": img_path.name
                    })
                    
        df = pd.DataFrame(data)
        self.metadata = df
        return df
    
    @abc.abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Perform analysis and return results
        
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save analysis results to cache directory"""
        if self.cache_dir:
            save_path = self.cache_dir / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, results)
            
    def load_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load cached results if available"""
        if self.cache_dir:
            load_path = self.cache_dir / filename
            if load_path.exists():
                return np.load(load_path, allow_pickle=True).item()
        return None