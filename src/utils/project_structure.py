# D:\3.Project\VGU\CS_AGE\face_analysis\src\utils\project_structure.py

import os
import json
from pathlib import Path
import logging
from datetime import datetime

class ProjectStructureManager:
    def __init__(self, base_dir):
        """
        Initialize project structure manager
        base_dir: Root directory of project (D:/3.Project/VGU/CS_AGE)
        """
        self.base_dir = Path(base_dir)
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_directory_structure(self, dataset_name="AFAD"):
        """Create directory structure for specified dataset"""
        directories = {
            f'face_analysis/data/processed/{dataset_name}': 'Processed dataset',
            f'face_analysis/data/metadata/{dataset_name}/stats': 'Dataset statistics',
            f'face_analysis/data/metadata/{dataset_name}/analysis': 'Analysis results',
        }

        # Create directories
        for dir_path, description in directories.items():
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f'Created directory: {full_path} - {description}')

    def setup_data_source(self):
        """Setup and validate data source configuration"""
        source = self.base_dir / 'data/AFAD'
        
        if not source.exists():
            self.logger.error(f"Source data not found at: {source}")
            return False

        # Tạo file config cho data source
        data_source = {
            'afad_source_path': str(source),
            'last_configured': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'active',
            'source_info': {
                'path': str(source),
                'size': sum(f.stat().st_size for f in source.rglob('*') if f.is_file()) / (1024*1024),  # MB
                'last_modified': datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Lưu thông tin vào metadata
        config_file = self.base_dir / 'face_analysis/data/metadata/data_source_config.json'
        with open(config_file, 'w') as f:
            json.dump(data_source, f, indent=4)
            
        self.logger.info(f"Configured to use AFAD data from: {source}")
        return True

    def validate_structure(self):
        """Validate the project structure"""
        # Kiểm tra source data
        source = self.base_dir / 'data/AFAD'
        if not source.exists():
            self.logger.error(f"Source data not found at: {source}")
            return False

        # Kiểm tra các thư mục processed và metadata
        required_paths = [
            'face_analysis/data/processed/AFAD',
            'face_analysis/data/metadata/AFAD/stats',
            'face_analysis/data/metadata/AFAD/analysis'
        ]

        missing_paths = []
        for path in required_paths:
            full_path = self.base_dir / path
            if not full_path.exists():
                missing_paths.append(path)

        if missing_paths:
            self.logger.error(f"Missing directories: {missing_paths}")
            return False

        self.logger.info("Project structure validation successful")
        return True

    def save_structure_info(self):
        """Save project structure information"""
        structure_info = {
            'project_root': str(self.base_dir),
            'directories': {
                'source_data': str(self.base_dir / 'data/AFAD'),
                'processed_data': str(self.base_dir / 'face_analysis/data/processed/AFAD'),
                'metadata': str(self.base_dir / 'face_analysis/data/metadata/AFAD')
            }
        }

        info_file = self.base_dir / 'face_analysis/data/metadata/project_structure.json'
        with open(info_file, 'w') as f:
            json.dump(structure_info, f, indent=4)

        self.logger.info(f"Saved structure info to: {info_file}")