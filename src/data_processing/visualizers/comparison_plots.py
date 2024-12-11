# face_analysis/src/data_processing/visualizers/comparison_plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

class ComparisonVisualizer:
    """Visualizer for comparing different analysis metrics"""
    
    def __init__(self, output_dir: str):
        """
        Initialize comparison visualizer
        
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

    def plot_quality_vs_age(self, quality_results: Dict[str, Any], 
                          distribution_results: Dict[str, Any],
                          save: bool = True) -> Optional[plt.Figure]:
        """Plot quality metrics versus age distribution"""
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # Age vs Image Quality
            gs = fig.add_gridspec(2, 2)
            
            # Brightness vs Age
            ax1 = fig.add_subplot(gs[0, 0])
            ages = distribution_results['age_distribution'].keys()
            brightness = quality_results['brightness_stats']['values']
            
            ax1.scatter(ages, brightness, alpha=0.5, c='orange')
            ax1.set_xlabel('Age')
            ax1.set_ylabel('Brightness')
            ax1.set_title('Brightness vs Age')
            
            # Contrast vs Age
            ax2 = fig.add_subplot(gs[0, 1])
            contrast = quality_results['contrast_stats']['values']
            
            ax2.scatter(ages, contrast, alpha=0.5, c='purple')
            ax2.set_xlabel('Age')
            ax2.set_ylabel('Contrast')
            ax2.set_title('Contrast vs Age')
            
            # Quality Distribution by Age Group
            ax3 = fig.add_subplot(gs[1, :])
            quality_by_age = self._compute_quality_by_age(
                quality_results, distribution_results
            )
            
            data = []
            categories = ['excellent', 'good', 'fair', 'poor']
            for age_group, scores in quality_by_age.items():
                for category in categories:
                    data.append({
                        'Age Group': age_group,
                        'Category': category,
                        'Percentage': scores[category]
                    })
                    
            df = pd.DataFrame(data)
            sns.barplot(
                x='Age Group',
                y='Percentage',
                hue='Category',
                data=df,
                ax=ax3
            )
            ax3.set_title('Quality Distribution by Age Group')
            ax3.set_xlabel('Age Group')
            ax3.set_ylabel('Percentage')
            
            plt.tight_layout()
            
            if save:
                save_path = self.output_dir / 'quality_vs_age_analysis.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved quality vs age plot to {save_path}")
                return None
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating quality vs age plot: {str(e)}")
            raise

    def plot_quality_vs_gender(self, quality_results: Dict[str, Any],
                             distribution_results: Dict[str, Any],
                             save: bool = True) -> Optional[plt.Figure]:
        """
        Plot quality metrics comparison between genders
        
        Args:
            quality_results: Results from quality analysis
            distribution_results: Results from distribution analysis
            save: Whether to save the plot to file
            
        Returns:
            matplotlib Figure if save=False, None otherwise
        """
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # Create subplots
            gs = fig.add_gridspec(2, 2)
            
            # 1. Overall Quality Distribution by Gender
            ax1 = fig.add_subplot(gs[0, :])
            
            # Prepare data
            categories = ['excellent', 'good', 'fair', 'poor']
            male_scores = [quality_results['quality_by_gender']['male'][cat] for cat in categories]
            female_scores = [quality_results['quality_by_gender']['female'][cat] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            # Create grouped bar plot
            rects1 = ax1.bar(x - width/2, male_scores, width, 
                           label='Male', color='lightblue', alpha=0.7)
            rects2 = ax1.bar(x + width/2, female_scores, width,
                           label='Female', color='lightpink', alpha=0.7)
            
            ax1.set_ylabel('Percentage (%)')
            ax1.set_title('Quality Distribution by Gender')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            
            # Add value labels
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax1.annotate(f'{height:.1f}%',
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            # 2. Technical Metrics by Gender Boxplots
            metrics = {
                'brightness': {'title': 'Brightness Distribution', 'color': 'gold'},
                'contrast': {'title': 'Contrast Distribution', 'color': 'purple'}
            }
            
            for idx, (metric, info) in enumerate(metrics.items()):
                ax = fig.add_subplot(gs[1, idx])
                
                # Get data by gender
                male_data = quality_results[f'{metric}_stats']['by_gender']['male']
                female_data = quality_results[f'{metric}_stats']['by_gender']['female']
                
                # Create violin plot with overlaid box plot
                parts = ax.violinplot(
                    [male_data, female_data],
                    showmeans=True,
                    showextrema=True
                )
                
                # Customize violin plot colors
                for pc in parts['bodies']:
                    pc.set_facecolor(info['color'])
                    pc.set_alpha(0.3)
                
                # Add box plot
                ax.boxplot(
                    [male_data, female_data],
                    positions=[1, 2],
                    widths=0.15,
                    showfliers=False
                )
                
                # Customize plot
                ax.set_title(info['title'])
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Male', 'Female'])
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = (
                    f"Male:\n"
                    f"μ={np.mean(male_data):.1f}\n"
                    f"σ={np.std(male_data):.1f}\n\n"
                    f"Female:\n"
                    f"μ={np.mean(female_data):.1f}\n"
                    f"σ={np.std(female_data):.1f}"
                )
                ax.text(0.02, 0.98, stats_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save:
                save_path = self.output_dir / 'quality_vs_gender_analysis.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved quality vs gender plot to {save_path}")
                return None
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating quality vs gender plot: {str(e)}")
            raise

    def create_comparison_report(self, quality_results: Dict[str, Any],
                               distribution_results: Dict[str, Any],
                               technical_results: Dict[str, Any],
                               save: bool = True) -> Optional[plt.Figure]:
        """
        Create comprehensive comparison analysis report
        
        Args:
            quality_results: Results from quality analysis
            distribution_results: Results from distribution analysis
            technical_results: Results from technical analysis
            save: Whether to save the plot to file
            
        Returns:
            matplotlib Figure if save=False, None otherwise
        """
        try:
            fig = plt.figure(figsize=(20, 15))
            
            # Create subplots
            gs = fig.add_gridspec(3, 2)
            
            # 1. Quality vs Age analysis
            ax1 = fig.add_subplot(gs[0, :])
            self.plot_quality_vs_age(quality_results, distribution_results, save=False)
            ax1.set_title('Quality Metrics by Age')
            
            # 2. Quality vs Gender analysis
            ax2 = fig.add_subplot(gs[1, :])
            self.plot_quality_vs_gender(quality_results, distribution_results, save=False)
            ax2.set_title('Quality Metrics by Gender')
            
            # 3. Technical metrics summary
            ax3 = fig.add_subplot(gs[2, :])
            self._plot_technical_summary(technical_results, ax3)
            
            plt.suptitle('Comprehensive Quality Comparison Analysis', fontsize=16)
            plt.tight_layout()
            
            if save:
                save_path = self.output_dir / 'comprehensive_comparison_report.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved comprehensive comparison report to {save_path}")
                return None
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating comparison report: {str(e)}")
            raise

    def _plot_technical_summary(self, technical_results: Dict[str, Any], ax: plt.Axes):
        """Helper method to plot technical metrics summary"""
        ax.axis('off')
        
        summary_text = [
            'Technical Metrics Summary:',
            f"Average File Size: {technical_results['size_analysis']['mean_size']/1024:.1f} KB",
            f"Most Common Format: {technical_results['format_analysis']['most_common']}",
            f"Color Channels: {technical_results['color_analysis']['channels']}",
            f"Average Resolution: {technical_results['resolution_analysis']['mean_resolution']}"
        ]
        
        ax.text(0.1, 0.7, '\n'.join(summary_text), fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))