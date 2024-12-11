# face_analysis/src/data_processing/reporters/report_generator.py

from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import shutil
from .report_templates import ReportTemplates

class ReportGenerator:
    """Handles report generation and file operations"""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for saving generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
            
    def _format_strengths(self, stats: Dict[str, Any]) -> str:
        """Format strengths section"""
        return f"""1. **Dataset Size:** Total of {stats['total_samples']:,} samples
2. **Age Coverage:** Spans {stats['age_range'][1] - stats['age_range'][0] + 1} years
3. **Data Completeness:** All samples have age and gender labels"""
    
    def _format_improvements(self, results: Dict[str, Any]) -> str:
        """Format improvements section"""
        return """1. **Age Distribution:**
   - Address underrepresented age groups
   - Balance sample counts across ages

2. **Gender Balance:**
   - Improve male/female ratio
   - Consider data augmentation strategies"""
    
    def _format_action_items(self, results: Dict[str, Any]) -> str:
        """Format action items section"""
        return """1. **Data Collection:**
   - Prioritize underrepresented age groups
   - Balance gender distribution

2. **Technical Approach:**
   - Implement stratified sampling
   - Consider weighted loss functions

3. **Monitoring:**
   - Track distribution metrics
   - Set up imbalance alerts"""
    
    def _copy_plots(self, report_dir: Path) -> None:
        """Copy plots to report directory"""
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            source_plots = self.output_dir.parent / "plots"
            if source_plots.exists():
                for plot_file in source_plots.glob("*.png"):
                    shutil.copy2(plot_file, plots_dir / plot_file.name)
        except Exception as e:
            self.logger.error(f"Error copying plots: {str(e)}")
    
    def generate_distribution_report(self, results: Dict[str, Any]) -> None:
        """Generate distribution analysis report"""
        try:
            # Create timestamped report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = self.output_dir / f"report_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy plots
            self._copy_plots(report_dir)
            
            # Prepare template data
            template_data = self._prepare_distribution_data(results)
            
            # Generate report content
            content = ReportTemplates.DISTRIBUTION_TEMPLATE.format(**template_data)
            
            # Save report
            report_path = report_dir / "distribution_analysis.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            self.logger.info(f"Distribution report generated at {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating distribution report: {str(e)}")
            raise
    
    def generate_report(self, title: str, results: Dict[str, Any]) -> None:
        """
        Generate a comprehensive markdown report
        
        Args:
            title: Report title
            results: Analysis results dictionary
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = self.output_dir / f"report_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy plots
            self._copy_plots(report_dir)
            
            # Generate report using template
            content = ""
            if "distribution_analysis" in results:
                distribution_results = results["distribution_analysis"]
                if distribution_results is not None and isinstance(distribution_results, dict):
                    try:
                        template_data = self._prepare_distribution_data(distribution_results)
                        content = ReportTemplates.DISTRIBUTION_TEMPLATE.format(**template_data)
                    except Exception as e:
                        self.logger.error(f"Error preparing distribution data: {str(e)}")
                        content = "Error occurred while processing distribution analysis results."
                else:
                    content = "Distribution analysis results are not available or invalid."
            else:
                content = "No analysis results available."
            
            # Save report
            report_path = report_dir / "analysis_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            self.logger.info(f"Report generated successfully at {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise

    def _prepare_distribution_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for distribution report template"""
        try:
            if not results or not isinstance(results, dict):
                raise ValueError("Invalid or empty results")
                
            required_keys = ['statistics', 'age_distribution', 'gender_distribution', 'imbalance_analysis']
            for key in required_keys:
                if key not in results:
                    raise ValueError(f"Missing required key: {key}")

            stats = results['statistics']
            age_dist = results['age_distribution']
            gender_dist = results['gender_distribution']
            imbalance = results['imbalance_analysis']

            # Safety checks
            if not age_dist:
                raise ValueError("Empty age distribution")
            if not gender_dist:
                raise ValueError("Empty gender distribution")

            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_samples': stats.get('total_samples', 0),
                'gender_ratio': stats.get('gender_ratio', 0.0),
                'age_mean': stats.get('age_mean', 0.0),
                'age_median': stats.get('age_median', 0.0),
                'age_std': stats.get('age_std', 0.0),
                'age_min': stats.get('age_range', [0, 0])[0],
                'age_max': stats.get('age_range', [0, 0])[1],
                'peak_age': max(age_dist, key=age_dist.get, default=0),
                'min_age': min(age_dist, key=age_dist.get, default=0),
                'male_count': gender_dist.get('male', 0),
                'female_count': gender_dist.get('female', 0),
                'age_entropy': imbalance.get('age', {}).get('entropy', 0.0),
                'age_norm_entropy': imbalance.get('age', {}).get('normalized_entropy', 0.0),
                'gender_entropy': imbalance.get('gender', {}).get('entropy', 0.0),
                'gender_norm_entropy': imbalance.get('gender', {}).get('normalized_entropy', 0.0),
                'strengths': self._format_strengths(stats),
                'improvements': self._format_improvements(results),
                'action_items': self._format_action_items(results)
            }
        except Exception as e:
            self.logger.error(f"Error preparing distribution data: {str(e)}")
            raise