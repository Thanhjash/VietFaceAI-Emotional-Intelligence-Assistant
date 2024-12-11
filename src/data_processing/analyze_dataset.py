# face_analysis/src/data_processing/analyze_dataset.py

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, TypedDict
import json
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from face_analysis.src.config.data_config import DATASET_CONFIGS
from face_analysis.src.data_processing.analyzers.distribution_analyzer import DistributionAnalyzer
from face_analysis.src.data_processing.visualizers.distribution_plots import DistributionVisualizer
from face_analysis.src.data_processing.reporters.report_generator import ReportGenerator

@dataclass
class AnalysisConfig:
    """Configuration for analysis process"""
    data_dir: Path
    output_dir: Path
    cache_dir: Optional[Path]
    dataset_name: str
    
class DistributionResults(TypedDict):
    """Type definition for distribution analysis results"""
    age_distribution: Dict[int, int]
    gender_distribution: Dict[str, int]
    age_gender_distribution: Dict[str, float]
    statistics: Dict[str, float]

class DataTransformer:
    """Handles data transformation between components"""
    
    @staticmethod
    def transform_age_gender_distribution(data: Dict) -> Dict[str, float]:
        """Transform age-gender distribution to correct format"""
        transformed = {}
        for age, gender_dict in data.items():
            if isinstance(gender_dict, dict):
                for gender, count in gender_dict.items():
                    key = f"({age}, '{gender}')"
                    transformed[key] = float(count)
        return transformed

    @staticmethod
    def validate_results(results: Dict[str, Any]) -> bool:
        """Validate analysis results structure"""
        required_keys = ['age_distribution', 'gender_distribution', 
                        'age_gender_distribution', 'statistics']
        return all(key in results for key in required_keys)

class DataAnalysisCoordinator:
    """Enhanced coordinator for the data analysis process"""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: Optional[str] = None,
        dataset_name: str = "AFAD"
    ):
        # Initialize configuration
        self.config = AnalysisConfig(
            data_dir=Path(data_dir),
            output_dir=Path(output_dir),
            cache_dir=Path(cache_dir) if cache_dir else None,
            dataset_name=dataset_name
        )
        
        # Setup components
        self._setup_components()
        self._setup_logging()
        
    def _setup_components(self):
        """Initialize analysis components"""
        # Create report directory
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = self.config.output_dir / f"report_{report_time}"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dataset config
        self.dataset_config = DATASET_CONFIGS[self.config.dataset_name]
        
        # Initialize analysis components
        self.distribution_analyzer = DistributionAnalyzer(
            str(self.config.data_dir),
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
            age_range=self.dataset_config["age_range"],
            gender_mapping=self.dataset_config["gender_mapping"]
        )
        
        self.distribution_visualizer = DistributionVisualizer(
            output_dir=str(self.report_dir / "plots")
        )
        
        self.report_generator = ReportGenerator(
            output_dir=str(self.report_dir)
        )
        
    def _setup_logging(self):
        """Setup enhanced logging configuration"""
        log_dir = self.config.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_distribution_analysis(self) -> DistributionResults:
        """Run enhanced distribution analysis and visualization"""
        self.logger.info("Starting distribution analysis...")
        
        try:
            # Run analysis
            results = self.distribution_analyzer.analyze()
            
            # Validate results
            if not DataTransformer.validate_results(results):
                raise ValueError("Invalid analysis results structure")
            
            # Debug logging
            self.logger.debug("Analysis results structure:")
            self.logger.debug(f"Keys: {results.keys()}")
            
            # Transform age-gender distribution
            results['age_gender_distribution'] = DataTransformer.transform_age_gender_distribution(
                results['age_gender_distribution']
            )
            
            # Generate visualizations with error handling
            try:
                self.distribution_visualizer.plot_age_distribution(results)
                self.logger.info("Age distribution plot generated successfully")
            except Exception as e:
                self.logger.error(f"Error generating age distribution plot: {str(e)}")
                
            try:
                self.distribution_visualizer.plot_gender_distribution(results)
                self.logger.info("Gender distribution plot generated successfully")
            except Exception as e:
                self.logger.error(f"Error generating gender distribution plot: {str(e)}")
                
            try:
                self.distribution_visualizer.plot_age_gender_distribution(results)
                self.logger.info("Age-gender distribution plot generated successfully")
            except Exception as e:
                self.logger.error(f"Error generating age-gender distribution plot: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in distribution analysis: {str(e)}")
            raise
        
    def run_full_analysis(self):
        """Run complete analysis pipeline with enhanced error handling"""
        self.logger.info("Starting full dataset analysis...")
        
        try:
            # Run distribution analysis
            distribution_results = self.run_distribution_analysis()
            
            # Generate report
            self.report_generator.generate_report(
                title=f"{self.config.dataset_name} Dataset Analysis Report",
                results={
                    "distribution_analysis": distribution_results
                }
            )
            
            self.logger.info("Full analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in full analysis: {str(e)}")
            raise
            
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "D:/3.Project/VGU/CS_AGE/data/AFAD"
    OUTPUT_DIR = "D:/3.Project/VGU/CS_AGE/face_analysis/data/analysis_output"
    CACHE_DIR = "D:/3.Project/VGU/CS_AGE/face_analysis/data/metadata"
    
    try:
        # Initialize and run coordinator
        coordinator = DataAnalysisCoordinator(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            cache_dir=CACHE_DIR,
            dataset_name="AFAD"
        )
        
        # Run analysis
        coordinator.run_full_analysis()
        
    except Exception as e:
        logging.error(f"Critical error in analysis: {str(e)}")
        sys.exit(1)