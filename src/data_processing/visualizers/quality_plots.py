# face_analysis/src/data_processing/visualizers/quality_plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class QualityVisualizer:
    """Visualizer for image quality analysis results"""
    
    def __init__(self, output_dir: str):
        """
        Initialize quality visualizer
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn')
        self.set_plot_style()
    
    def set_plot_style(self):
        """Configure common plot styling"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.titlesize': 20,
            'figure.figsize': (12, 8),
            'figure.dpi': 100,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def plot_resolution_distribution(self, results: Dict[str, Any], save: bool = True) -> Optional[plt.Figure]:
        """Plot resolution distribution analysis"""
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # Create subplots
            gs = fig.add_gridspec(2, 2)
            
            # Resolution scatter plot
            ax1 = fig.add_subplot(gs[0, :])
            widths = results['resolution_stats']['widths']
            heights = results['resolution_stats']['heights']
            
            scatter = ax1.scatter(widths, heights, alpha=0.5, c=np.array(widths)/np.array(heights),
                                cmap='viridis')
            ax1.set_xlabel('Width (pixels)')
            ax1.set_ylabel('Height (pixels)')
            ax1.set_title('Image Resolutions Distribution')
            plt.colorbar(scatter, label='Aspect Ratio')
            
            # Aspect ratio histogram
            ax2 = fig.add_subplot(gs[1, 0])
            aspects = results['resolution_stats']['aspects']
            ax2.hist(aspects, bins=50, color='skyblue', alpha=0.7)
            ax2.set_xlabel('Aspect Ratio')
            ax2.set_ylabel('Count')
            ax2.set_title('Aspect Ratio Distribution')
            
            # Resolution statistics
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.axis('off')
            stats_text = [
                'Resolution Statistics:',
                f"Mean Width: {results['resolution_stats']['widths_stats']['mean']:.1f}px",
                f"Mean Height: {results['resolution_stats']['heights_stats']['mean']:.1f}px",
                f"Mean Aspect Ratio: {results['resolution_stats']['aspects_stats']['mean']:.2f}",
                f"Most Common Resolution:",
                f"  {int(max(set(widths), key=widths.count))}x{int(max(set(heights), key=heights.count))}px"
            ]
            ax3.text(0.1, 0.7, '\n'.join(stats_text), fontsize=12)
            
            plt.tight_layout()
            
            if save:
                save_path = self.output_dir / 'resolution_analysis.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved resolution analysis plot to {save_path}")
                return None
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating resolution plot: {str(e)}")
            raise

    def plot_quality_metrics(self, results: Dict[str, Any], save: bool = True) -> Optional[plt.Figure]:
        """Plot quality metrics analysis"""
        try:
            fig = plt.figure(figsize=(15, 12))
            
            # Create subplots
            gs = fig.add_gridspec(3, 2)
            
            # Brightness distribution
            ax1 = fig.add_subplot(gs[0, 0])
            brightness_values = results['brightness_stats']['values']
            ax1.hist(brightness_values, bins=50, color='gold', alpha=0.7)
            ax1.axvline(results['brightness_stats']['statistics']['mean'], 
                       color='red', linestyle='--', label='Mean')
            ax1.set_xlabel('Brightness Value')
            ax1.set_ylabel('Count')
            ax1.set_title('Brightness Distribution')
            ax1.legend()
            
            # Contrast distribution
            ax2 = fig.add_subplot(gs[0, 1])
            contrast_values = results['contrast_stats']['values']
            ax2.hist(contrast_values, bins=50, color='purple', alpha=0.7)
            ax2.axvline(results['contrast_stats']['statistics']['mean'],
                       color='red', linestyle='--', label='Mean')
            ax2.set_xlabel('Contrast Value')
            ax2.set_ylabel('Count')
            ax2.set_title('Contrast Distribution')
            ax2.legend()
            
            # Blur score distribution
            ax3 = fig.add_subplot(gs[1, 0])
            blur_scores = results['blur_stats']['scores']
            ax3.hist(blur_scores, bins=50, color='green', alpha=0.7)
            ax3.axvline(results['blur_stats']['statistics']['mean'],
                       color='red', linestyle='--', label='Mean')
            ax3.set_xlabel('Blur Score')
            ax3.set_ylabel('Count')
            ax3.set_title('Blur Score Distribution')
            ax3.legend()
            
            # Overall quality pie chart
            ax4 = fig.add_subplot(gs[1, 1])
            quality_scores = results['overall_quality']
            colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c']
            ax4.pie(quality_scores.values(), labels=quality_scores.keys(),
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax4.set_title('Overall Quality Distribution')
            
            # Quality metrics summary
            ax5 = fig.add_subplot(gs[2, :])
            ax5.axis('off')
            summary_text = [
                'Quality Metrics Summary:',
                f"Mean Brightness: {results['brightness_stats']['statistics']['mean']:.2f}",
                f"Mean Contrast: {results['contrast_stats']['statistics']['mean']:.2f}",
                f"Mean Blur Score: {results['blur_stats']['statistics']['mean']:.2f}",
                f"Potential Blurry Images: {results['blur_stats']['statistics']['potential_blur_count']}",
                f"Excellent Quality Images: {quality_scores['excellent']:.1f}%",
                f"Poor Quality Images: {quality_scores['poor']:.1f}%"
            ]
            ax5.text(0.1, 0.7, '\n'.join(summary_text), fontsize=12)
            
            plt.tight_layout()
            
            if save:
                save_path = self.output_dir / 'quality_metrics_analysis.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved quality metrics plot to {save_path}")
                return None
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating quality metrics plot: {str(e)}")
            raise

    def create_quality_report(self, results: Dict[str, Any], save: bool = True) -> Optional[plt.Figure]:
        """Create comprehensive quality analysis report"""
        try:
            # Create all plots
            fig = plt.figure(figsize=(20, 15))
            
            # Resolution analysis
            plt.subplot(221)
            self.plot_resolution_distribution(results, save=False)
            
            # Quality metrics
            plt.subplot(222)
            self.plot_quality_metrics(results, save=False)
            
            plt.suptitle('Image Quality Analysis Report', fontsize=16)
            plt.tight_layout()
            
            if save:
                save_path = self.output_dir / 'quality_analysis_report.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved quality analysis report to {save_path}")
                return None
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating quality report: {str(e)}")
            raise