# face_analysis/src/data_processing/visualizers/distribution_plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
import pandas as pd
from contextlib import contextmanager

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    font_size: int = 12
    title_size: int = 16
    label_size: int = 14
    grid_alpha: float = 0.3

class DataValidator:
    """Validates input data for visualization"""
    
    @staticmethod
    def validate_distribution_data(data: Dict[str, Any]) -> bool:
        """Validate required data fields exist and have correct format"""
        required_fields = ['age_distribution', 'gender_distribution', 'age_gender_distribution', 'statistics']
        return all(field in data for field in required_fields)

class DataTransformer:
    """Transforms data for visualization"""
    
    @staticmethod
    def transform_age_gender_distribution(data: Dict) -> pd.DataFrame:
        """Transform age-gender distribution to standardized DataFrame"""
        parsed_data = []
        
        for key, value in data.items():
            try:
                # Handle string tuple format like "(male, '25')"
                if isinstance(key, str):
                    # Remove parentheses and split
                    clean_key = key.strip("()' ")
                    parts = [part.strip("' ") for part in clean_key.split(',')]
                    if len(parts) == 2:
                        gender, age = parts[0], parts[1]
                elif isinstance(key, tuple):
                    gender, age = key
                else:
                    continue

                # Convert age to int and ensure gender is string
                try:
                    age = int(age)
                    gender = str(gender)
                    count = float(value)
                    
                    parsed_data.append({
                        'age': age,
                        'gender': gender,
                        'count': count
                    })
                except (ValueError, TypeError) as e:
                    logging.warning(f"Could not parse value for {key}: {str(e)}")
                    continue
                    
            except Exception as e:
                logging.warning(f"Error processing key-value pair: {key}-{value}. Error: {str(e)}")
                continue
        
        return pd.DataFrame(parsed_data)

class DistributionVisualizer:
    """Enhanced visualizer for creating statistical distribution plots"""
    
    def __init__(self, output_dir: str):
        """Initialize visualizer with output directory and style settings"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.config = VisualizationConfig()
        
        # Configure plot style
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Configure matplotlib plot styling"""
        plt.style.use('seaborn')
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.labelsize': self.config.label_size,
            'axes.titlesize': self.config.title_size,
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'axes.grid': True,
            'grid.alpha': self.config.grid_alpha
        })

    @contextmanager
    def _plot_context(self, filename: str):
        """Context manager for plot creation and saving"""
        try:
            yield
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Successfully saved plot to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving plot {filename}: {str(e)}")
            raise
        finally:
            plt.close()

    def plot_age_gender_distribution(self, results: Dict[str, Any]) -> None:
        """Create enhanced age-gender distribution heatmap"""
        try:
            # Validate input data
            if not DataValidator.validate_distribution_data(results):
                self.logger.error("Invalid input data format")
                return

            # Transform data
            df = DataTransformer.transform_age_gender_distribution(
                results['age_gender_distribution']
            )
            
            if df.empty:
                self.logger.error("No valid data points after transformation")
                return

            # Create pivot table for heatmap
            pivot_table = pd.pivot_table(
                df,
                values='count',
                index='age',
                columns='gender',
                aggfunc='sum',
                fill_value=0
            )

            # Sort index for better visualization
            pivot_table.sort_index(inplace=True)

            with self._plot_context('age_gender_distribution_detailed.png'):
                plt.figure(figsize=(14, 8))
                
                # Create heatmap
                sns.heatmap(
                    pivot_table,
                    cmap='YlOrRd',
                    annot=True,
                    fmt='.0f',
                    cbar_kws={'label': 'Sample Count'}
                )
                
                plt.title('Age-Gender Distribution Heatmap')
                plt.xlabel('Gender')
                plt.ylabel('Age')
                plt.tight_layout()

        except Exception as e:
            self.logger.error(f"Error in plot_age_gender_distribution: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)
            return

    def plot_gender_distribution(self, results: Dict[str, Any]) -> None:
        """Create gender distribution visualization with statistics"""
        try:
            if not DataValidator.validate_distribution_data(results):
                self.logger.error("Invalid input data format")
                return
                
            gender_dist = results['gender_distribution']
            if not gender_dist:
                self.logger.error("Empty gender distribution data")
                return

            with self._plot_context('gender_distribution_detailed.png'):
                fig = plt.figure(figsize=(12, 8))
                
                # Create main pie chart
                ax1 = plt.subplot(121)
                
                total = sum(gender_dist.values())
                gender_pcts = {k: (v/total)*100 for k, v in gender_dist.items()}
                
                # Create pie chart
                wedges, texts, autotexts = ax1.pie(
                    gender_dist.values(),
                    explode=[0.05] * len(gender_dist),
                    labels=gender_dist.keys(),
                    autopct='%1.1f%%',
                    colors=['lightblue', 'lightpink'],
                    shadow=True,
                    startangle=90
                )
                
                ax1.axis('equal')
                ax1.set_title('Gender Distribution')
                
                # Add statistics table
                ax2 = plt.subplot(122)
                ax2.axis('off')
                
                table_data = [
                    ['Metric', 'Value'],
                    ['Total Samples', str(total)],
                    *[
                        [f'{gender.title()} Count', 
                         f"{count} ({gender_pcts[gender]:.1f}%)"]
                        for gender, count in gender_dist.items()
                    ],
                    ['Gender Ratio (M/F)', f"{results['statistics'].get('gender_ratio', 'N/A'):.2f}"]
                ]
                
                table = ax2.table(
                    cellText=table_data,
                    loc='center',
                    cellLoc='left',
                    colWidths=[0.4, 0.6]
                )
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 2)

                plt.tight_layout()

        except Exception as e:
            self.logger.error(f"Error creating gender distribution plot: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)
            raise

    def plot_age_distribution(self, results: Dict[str, Any]) -> None:
        """Create age distribution visualization with detailed statistics"""
        try:
            if not DataValidator.validate_distribution_data(results):
                self.logger.error("Invalid input data format")
                return
                
            age_dist = results['age_distribution']
            if not age_dist:
                self.logger.error("Empty age distribution data")
                return

            with self._plot_context('age_distribution_detailed.png'):
                fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(14, 10))
                
                ages = sorted(age_dist.keys())
                counts = [age_dist[age] for age in ages]
                
                # Main distribution plot
                ax1.bar(ages, counts, alpha=0.6, color='skyblue', label='Sample Count')
                
                # Add trend line if enough data points
                if len(ages) > 2:
                    z = np.polyfit(ages, counts, 2)
                    p = np.poly1d(z)
                    ax1.plot(ages, p(ages), "r--", alpha=0.8, label='Trend')
                
                # Add statistics lines
                stats = results.get('statistics', {})
                mean_age = stats.get('age_mean')
                median_age = stats.get('age_median')
                
                if mean_age is not None:
                    ax1.axvline(mean_age, color='g', linestyle='--', 
                              label=f'Mean ({mean_age:.1f})')
                if median_age is not None:
                    ax1.axvline(median_age, color='y', linestyle='--',
                              label=f'Median ({median_age:.1f})')
                
                ax1.set_title('Age Distribution in Dataset')
                ax1.set_xlabel('Age')
                ax1.set_ylabel('Number of Samples')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Create boxplot data
                boxplot_data = []
                for age, count in age_dist.items():
                    boxplot_data.extend([age] * count)
                
                if boxplot_data:
                    ax2.boxplot(boxplot_data, vert=False, showfliers=False)
                    ax2.set_title('Age Distribution Box Plot')
                
                plt.tight_layout()

        except Exception as e:
            self.logger.error(f"Error creating age distribution plot: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)
            raise