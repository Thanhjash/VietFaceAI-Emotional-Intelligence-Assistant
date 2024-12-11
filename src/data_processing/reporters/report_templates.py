# face_analysis/src/data_processing/reporters/report_templates.py

class ReportTemplates:
    """Static templates for different types of analysis reports"""
    
    DISTRIBUTION_TEMPLATE = """# AFAD Dataset Distribution Analysis
    
## 1. Overview
- **Analysis Date:** {timestamp}
- **Total Samples:** {total_samples:,}
- **Gender Ratio (M/F):** {gender_ratio:.2f}

## 2. Age Analysis
### 2.1 Basic Statistics
- **Mean Age:** {age_mean:.2f} years
- **Median Age:** {age_median:.2f} years
- **Standard Deviation:** {age_std:.2f}
- **Age Range:** {age_min} - {age_max} years

### 2.2 Age Distribution
![Age Distribution](plots/age_distribution_detailed.png)

**Key Observations:**
- Peak age group: {peak_age}
- Least represented age: {min_age}

## 3. Gender Analysis
### 3.1 Gender Ratio
![Gender Distribution](plots/gender_distribution_detailed.png)

### 3.2 Detailed Counts
- **Male:** {male_count:,} samples
- **Female:** {female_count:,} samples

## 4. Age-Gender Analysis
### 4.1 Distribution Heatmap
![Age-Gender Distribution](plots/age_gender_distribution_detailed.png)

### 4.2 Imbalance Metrics
- **Age:**
  - Entropy: {age_entropy:.3f}
  - Normalized Entropy: {age_norm_entropy:.3f}
- **Gender:**
  - Entropy: {gender_entropy:.3f}
  - Normalized Entropy: {gender_norm_entropy:.3f}

## 5. Evaluation and Recommendations

### 5.1 Strengths
{strengths}

### 5.2 Areas for Improvement
{improvements}

### 5.3 Action Items
{action_items}
"""

    QUALITY_TEMPLATE = """# AFAD Dataset Quality Analysis

## 1. Image Quality Metrics
### 1.1 Resolution Analysis
{resolution_section}

### 1.2 Quality Scores
{quality_section}

## 2. Technical Characteristics
{technical_section}

## 3. Recommendations
{recommendations}
"""

    TECHNICAL_TEMPLATE = """# AFAD Dataset Technical Analysis

## 1. Image Characteristics
### 1.1 Format Distribution
{format_section}

### 1.2 Size Analysis
{size_section}

### 1.3 Color Properties
{color_section}

## 2. Technical Recommendations
{recommendations}
"""

    STATISTICAL_TEMPLATE = """# AFAD Dataset Statistical Analysis

## 1. Descriptive Statistics
{descriptive_stats}

## 2. Distribution Analysis
{distribution_stats}

## 3. Correlation Analysis
{correlation_stats}

## 4. Statistical Insights
{insights}
"""

    FULL_REPORT_TEMPLATE = """# AFAD Dataset Comprehensive Analysis Report

{distribution_analysis}

---
{quality_analysis}

---
{technical_analysis}

---
{statistical_analysis}
"""