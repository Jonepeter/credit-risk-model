"""
Exploratory Data Analysis (EDA) - Credit Risk Model

This script performs comprehensive exploratory data analysis on the credit risk dataset
to uncover patterns, identify data quality issues, and form hypotheses for feature engineering.

Objectives:
1. Understand the structure and characteristics of the dataset
2. Identify data quality issues (missing values, outliers)
3. Analyze distributions of numerical and categorical features
4. Explore correlations between features
5. Form hypotheses for feature engineering

Key Insights to Extract:
- Top 3-5 most important insights from the data
- Data quality issues and their implications
- Feature relationships and potential engineering opportunities
- Risk factors and their distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import datetime
import os

# Visualization settings
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# --- Configuration for plots ---
sns.set_style("whitegrid") # A nice background style for seaborn plots
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size
plt.rcParams['font.size'] = 12 # Default font size

# Display settings for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

def load_data(data_path):
    """Load the dataset or create a sample dataset for demonstration."""
    """
    Load the dataset from the specified path.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the credit risk dataset.

    Returns:
    --------
    pandas.DataFrame
        Loaded dataset with transaction data for credit risk analysis.
        If file not found, returns a sample dataset with similar structure.
    """
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"----------------- Dataset loaded successfully! --------------------")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Data file not found. Creating a sample dataset for demonstration purposes...")


def display_missing_values(df):
    """
    Display missing values count and percentage for all columns
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print(f"\nMissing Values Summary:")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df) * 100)
    
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_percentage
    }).sort_values('Missing_Percentage', ascending=False)
    
    # Only show columns with missing values
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    
    if len(missing_summary) > 0:
        for col in missing_summary.index:
            count = missing_summary.loc[col, 'Missing_Count']
            pct = missing_summary.loc[col, 'Missing_Percentage']
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("  No missing values found in any column.")
        
def analysis_numerical_distributions(df, numerical_cols=None, figsize=(15, 10)):
    """
    Plot distribution of numerical features using histograms and box plots
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_cols (list): List of numerical columns to plot. If None, auto-detect
        figsize (tuple): Figure size for the plot
    """
    if numerical_cols == None:
        numerical_cols = df.select_dtypes(include = [np.number]).columns.to_list()
    else:
        print("There are no numerical columns.")
    for i, feature in enumerate(numerical_cols):
        plt.figure(figsize=(12, 6))

        # Calculate and print skewness
        skewness = df[feature].skew()
        print(f"Skewness for {feature}: {skewness:.2f}")
        if skewness > 0.5:
            print(f"  --> {feature} is positively skewed (tail to the right).")
        elif skewness < -0.5:
            print(f"  --> {feature} is negatively skewed (tail to the left).")
        else:
            print(f"  --> {feature} is relatively symmetric.")

        # Histogram
        plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}', fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        # Box Plot for Outlier Detection
        plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
        sns.boxplot(y=df[feature])
        plt.title(f'Box Plot of {feature}', fontsize=14)
        plt.ylabel(feature, fontsize=12)

        plt.tight_layout() # Adjusts plot parameters for a tight layout
        plt.show()
    

    # Check for outliers using IQR method
    print(f"\nOutlier Analysis (IQR Method):")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_percentage = (len(outliers) / len(df[col].dropna())) * 100
        
        print(f"  {col}: {len(outliers)} outliers ({outlier_percentage:.2f}%)")

def handle_skewness(df, numerical_cols=None, threshold=0.5):
    """
    Analyze and provide recommendations for handling skewed features
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_cols (list): List of numerical columns to analyze
        threshold (float): Skewness threshold for classification
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("=" * 60)
    print("SKEWNESS ANALYSIS & RECOMMENDATIONS")
    print("=" * 60)
    
    skewness_results = {}
    
    for col in numerical_cols:
        skewness = df[col].skew()
        skewness_results[col] = skewness
        
        print(f"\n{col}:")
        print(f"  Skewness: {skewness:.2f}")
        
        if skewness > threshold:
            print(f"  Type: POSITIVE SKEW")
            print(f"  Recommendations:")
            print(f"    - Apply log transformation: np.log1p(df['{col}'])")
            print(f"    - Use RobustScaler instead of StandardScaler")
            print(f"    - Consider winsorization for outliers")
            print(f"    - Models: Random Forest, XGBoost handle this well")
            
        elif skewness < -threshold:
            print(f"  Type: NEGATIVE SKEW")
            print(f"  Recommendations:")
            print(f"    - Apply power transformation: df['{col}'] ** 2")
            print(f"    - Focus on left tail for risk assessment")
            print(f"    - Create binary flags for extreme low values")
            print(f"    - Use percentile-based binning")
            
        else:
            print(f"  Type: RELATIVELY SYMMETRIC")
            print(f"  Recommendations:")
            print(f"    - No transformation needed")
            print(f"    - Standard preprocessing works fine")
            print(f"    - Linear models (Logistic Regression) suitable")
            print(f"    - StandardScaler appropriate")
    
    return skewness_results

def apply_skewness_transformations(df, numerical_cols=None, threshold=0.5):
    """
    Apply appropriate transformations based on skewness analysis
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_cols (list): List of numerical columns to transform
        threshold (float): Skewness threshold for transformation
    
    Returns:
        pd.DataFrame: DataFrame with transformed features
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_transformed = df.copy()
    
    print("\n" + "=" * 60)
    print("APPLYING SKEWNESS TRANSFORMATIONS")
    print("=" * 60)
    
    for col in numerical_cols:
        skewness = df[col].skew()
        threshold = determine_skewness_threshold(df)
        if skewness > threshold:
            # Positive skew - apply log transformation
            if df[col].min() >= 0:
                df_transformed[f'{col}_log'] = np.log1p(df[col])
                print(f"Applied log transformation to {col}")
            else:
                # Handle negative values
                min_val = df[col].min()
                df_transformed[f'{col}_log'] = np.log1p(df[col] - min_val + 1)
                print(f"Applied log transformation to {col} (adjusted for negative values)")
                
        elif skewness < -threshold:
            # Negative skew - apply power transformation
            df_transformed[f'{col}_squared'] = df[col] ** 2
            print(f"Applied square transformation to {col}")
    
    return df_transformed

def determine_skewness_threshold(df, model_type='auto', dataset_size=None):
    """
    Determine optimal skewness threshold based on dataset and model characteristics
    
    Args:
        df (pd.DataFrame): Input dataframe
        model_type (str): Type of model to be used ('linear', 'tree', 'auto')
        dataset_size (int): Number of samples in dataset
    
    Returns:
        float: Recommended threshold value
    """
    if dataset_size is None:
        dataset_size = len(df)
    
    print("=" * 60)
    print("SKEWNESS THRESHOLD RECOMMENDATION")
    print("=" * 60)
    
    # Analyze current skewness distribution
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skewness_values = [df[col].skew() for col in numerical_cols]
    
    print(f"Dataset size: {dataset_size:,} samples")
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Current skewness range: {min(skewness_values):.2f} to {max(skewness_values):.2f}")
    print(f"Mean skewness: {np.mean(skewness_values):.2f}")
    
    # Determine threshold based on criteria
    if model_type == 'linear':
        # Linear models are sensitive to skewness
        if dataset_size < 1000:
            threshold = 0.5
            reasoning = "Small dataset + Linear model = Conservative approach"
        elif dataset_size < 10000:
            threshold = 0.7
            reasoning = "Medium dataset + Linear model = Moderate approach"
        else:
            threshold = 1.0
            reasoning = "Large dataset + Linear model = Standard approach"
            
    elif model_type == 'tree':
        # Tree-based models are robust to skewness
        if dataset_size < 1000:
            threshold = 1.0
            reasoning = "Small dataset + Tree model = Moderate approach"
        elif dataset_size < 10000:
            threshold = 1.5
            reasoning = "Medium dataset + Tree model = Liberal approach"
        else:
            threshold = 2.0
            reasoning = "Large dataset + Tree model = Very liberal approach"
            
    else:  # 'auto' - determine based on data characteristics
        # Check if data is naturally skewed (common in financial data)
        high_skew_count = sum(1 for skew in skewness_values if abs(skew) > 1.0)
        skew_percentage = (high_skew_count / len(skewness_values)) * 100
        
        if skew_percentage > 50:
            # Many features are naturally skewed (common in financial data)
            threshold = 1.5
            reasoning = f"High natural skewness ({skew_percentage:.1f}% features > 1.0) - Liberal approach"
        elif dataset_size < 1000:
            threshold = 0.5
            reasoning = "Small dataset - Conservative approach"
        elif dataset_size < 10000:
            threshold = 1.0
            reasoning = "Medium dataset - Standard approach"
        else:
            threshold = 1.5
            reasoning = "Large dataset - Liberal approach"
    
    print(f"\nRecommended threshold: {threshold}")
    print(f"Reasoning: {reasoning}")
    
    # Show impact of this threshold
    features_above_threshold = sum(1 for skew in skewness_values if abs(skew) > threshold)
    print(f"\nImpact of this threshold:")
    print(f"  Features requiring transformation: {features_above_threshold}/{len(numerical_cols)} ({features_above_threshold/len(numerical_cols)*100:.1f}%)")
    
    # Alternative thresholds for comparison
    print(f"\nAlternative thresholds for comparison:")
    for alt_threshold in [0.5, 1.0, 1.5, 2.0]:
        if alt_threshold != threshold:
            count = sum(1 for skew in skewness_values if abs(skew) > alt_threshold)
            print(f"  {alt_threshold}: {count}/{len(numerical_cols)} features ({count/len(numerical_cols)*100:.1f}%)")
    
    return threshold

def analyze_skewness_impact(df, thresholds=[0.5, 1.0, 1.5, 2.0]):
    """
    Analyze the impact of different skewness thresholds
    
    Args:
        df (pd.DataFrame): Input dataframe
        thresholds (list): List of thresholds to compare
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skewness_values = [df[col].skew() for col in numerical_cols]
    
    print("=" * 60)
    print("SKEWNESS THRESHOLD IMPACT ANALYSIS")
    print("=" * 60)
    
    print(f"Feature\t\tSkewness\tThresholds that would transform:")
    print("-" * 70)
    
    for col, skew in zip(numerical_cols, skewness_values):
        transformations = []
        for threshold in thresholds:
            if abs(skew) > threshold:
                transformations.append(str(threshold))
        
        if transformations:
            print(f"{col:<15}\t{skew:>7.2f}\t{', '.join(transformations)}")
    
    print(f"\nSummary by threshold:")
    for threshold in thresholds:
        count = sum(1 for skew in skewness_values if abs(skew) > threshold)
        percentage = (count / len(numerical_cols)) * 100
        print(f"  {threshold}: {count}/{len(numerical_cols)} features ({percentage:.1f}%)")


def analyze_categorical_distributions(df, categorical_cols=None, max_categories=10, figsize=(15, 10)):
    """
    Analyze and visualize the distribution of categorical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_cols (list): List of categorical columns to analyze. If None, auto-detect
        max_categories (int): Maximum number of categories to display in plots
        figsize (tuple): Figure size for the plot
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found in the dataset.")
        return
    
    print(f"\n{'='*60}")
    print("CATEGORICAL FEATURES DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    for i, feature in enumerate(categorical_cols):
        print(f"\n--- {feature} ---")
        
        # Basic statistics
        unique_count = df[feature].nunique()
        missing_count = df[feature].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        print(f"Unique categories: {unique_count}")
        print(f"Missing values: {missing_count} ({missing_pct:.2f}%)")
        
        # Value counts
        value_counts = df[feature].value_counts()
        print(f"\nTop {min(max_categories, len(value_counts))} categories:")
        for j, (category, count) in enumerate(value_counts.head(max_categories).items()):
            percentage = (count / len(df)) * 100
            print(f"  {j+1}. {category}: {count} ({percentage:.2f}%)")
        
        if len(value_counts) > max_categories:
            remaining = len(value_counts) - max_categories
            remaining_count = value_counts.iloc[max_categories:].sum()
            remaining_pct = (remaining_count / len(df)) * 100
            print(f"  ... and {remaining} more categories: {remaining_count} ({remaining_pct:.2f}%)")
        
        # Additional insights
        print(f"\nInsights for {feature}:")
        
        # Check for class imbalance
        most_common_pct = (value_counts.iloc[0] / len(df)) * 100
        if most_common_pct > 80:
            print(f"  ⚠️  High class imbalance: Most common category is {most_common_pct:.1f}%")
        elif most_common_pct > 60:
            print(f"  ⚠️  Moderate class imbalance: Most common category is {most_common_pct:.1f}%")
        else:
            print(f"  ✅ Good category balance: Most common category is {most_common_pct:.1f}%")
        
        # Check for rare categories
        rare_categories = value_counts[value_counts < len(df) * 0.01]  # Less than 1%
        if len(rare_categories) > 0:
            print(f"  ⚠️  {len(rare_categories)} rare categories (<1% each) found")
        
        # Check for high cardinality
        if unique_count > 50:
            print(f"  ⚠️  High cardinality: {unique_count} unique categories")
        elif unique_count > 20:
            print(f"  ⚠️  Moderate cardinality: {unique_count} unique categories")
        else:
            print(f"  ✅ Low cardinality: {unique_count} unique categories")
        
        print("-" * 40)

def get_categorical_summary(df, categorical_cols=None):
    """
    Get a summary of all categorical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_cols (list): List of categorical columns to analyze. If None, auto-detect
    
    Returns:
        pd.DataFrame: Summary statistics for categorical features
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    summary_data = []
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        most_common = value_counts.index[0] if len(value_counts) > 0 else None
        most_common_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
        most_common_pct = (most_common_count / len(df)) * 100
        
        summary_data.append({
            'Column': col,
            'Unique_Count': unique_count,
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct,
            'Most_Common': most_common,
            'Most_Common_Count': most_common_count,
            'Most_Common_Percentage': most_common_pct,
            'Cardinality_Level': 'High' if unique_count > 50 else 'Moderate' if unique_count > 20 else 'Low',
            'Balance_Level': 'Imbalanced' if most_common_pct > 60 else 'Moderate' if most_common_pct > 40 else 'Balanced'
        })
    
    return pd.DataFrame(summary_data)

def correlation_analysis(dataset):
    # Correlation Analysis for Numerical Features
    print("=" * 60)
    print("CORRELATION ANALYSIS".center(60))
    print("=" * 60)

    # Get numerical columns (excluding target variable for now)
    numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Analyzing correlations for {len(numerical_cols)} numerical features:")
    print(f"Features: {numerical_cols}")

    # Calculate correlation matrix
    correlation_matrix = dataset[numerical_cols].corr()

    # Find highly correlated features
    print("\n" + "=" * 60)
    print("HIGHLY CORRELATED FEATURES ANALYSIS")
    print("=" * 60)


    # Find pairs with high correlation
    high_corr_threshold = 0.8
    moderate_corr_threshold = 0.5

    high_corr_pairs = []
    moderate_corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= high_corr_threshold:
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                    correlation_matrix.columns[j], 
                                    corr_value))
            elif abs(corr_value) >= moderate_corr_threshold:
                moderate_corr_pairs.append((correlation_matrix.columns[i], 
                                        correlation_matrix.columns[j], 
                                        corr_value))

    # Display high correlations
    if high_corr_pairs:
        print(f"\n🔴 HIGH CORRELATIONS (≥{high_corr_threshold}):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
            print(f"    → Consider removing one feature to avoid multicollinearity")
    else:
        print(f"\n✅ No high correlations (≥{high_corr_threshold}) found")

    # Display moderate correlations
    if moderate_corr_pairs:
        print(f"\n🟡 MODERATE CORRELATIONS (≥{moderate_corr_threshold}):")
        for feat1, feat2, corr in moderate_corr_pairs:
            print(f"  {feat1} ↔ {feat2}: {corr:.3f}")

    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

    # Correlation insights and recommendations
    print("\n" + "=" * 60)
    print("CORRELATION INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    if high_corr_pairs:
        print("  • High multicollinearity detected - feature selection needed")
        print("  • Consider PCA or feature elimination")
    else:
        print("  • No severe multicollinearity issues")


def detect_outliers(df, numerical_features=None, outlier_threshold=1.5):
    """
    Detect outliers in numerical features using box plots and IQR method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    numerical_features : list, optional
        List of numerical feature names to analyze. If None, all numerical features are used.
    outlier_threshold : float, default=1.5
        Multiplier for IQR to define outlier bounds (default: 1.5)
    
    Returns:
    --------
    dict
        Dictionary containing outlier statistics for each feature
    
    Raises:
    -------
    ValueError
        If df is empty or contains no numerical features
    TypeError
        If df is not a pandas DataFrame
    """
    try:
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Get numerical features if not specified
        if numerical_features is None:
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_features:
            raise ValueError("No numerical features found in the DataFrame")
        
        print("\n" + "=" * 60)
        print("OUTLIER DETECTION ANALYSIS")
        print("=" * 60)
        
        # Calculate subplot layout
        n_features = len(numerical_features)
        cols = 3
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        outlier_summary = {}
        
        for idx, feature in enumerate(numerical_features):
            try:
                row = idx // cols
                col = idx % cols
                
                # Check if feature exists
                if feature not in df.columns:
                    print(f"⚠️  Warning: Feature '{feature}' not found in DataFrame")
                    continue
                
                # Create box plot
                axes[row, col].boxplot(df[feature].dropna())
                axes[row, col].set_title(f'Box Plot: {feature}')
                axes[row, col].set_ylabel('Values')
            
            except Exception as e:
                print(f"⚠️  Error processing feature '{feature}': {str(e)}")
                continue
        
        # Hide empty subplots
        for idx in range(n_features, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Print outlier summary
        print("\n📊 OUTLIER SUMMARY:")
        print("-" * 40)
        for feature, stats in outlier_summary.items():
            if stats['count'] > 0:
                print(f"🔴 {feature}:")
                print(f"   • Outliers: {stats['count']} ({stats['percentage']:.1f}%)")
                print(f"   • Range: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
            else:
                print(f"✅ {feature}: No outliers detected")
        
        # Recommendations
        print("\n💡 OUTLIER HANDLING RECOMMENDATIONS:")
        print("-" * 40)
        high_outlier_features = [f for f, stats in outlier_summary.items() if stats['percentage'] > 5]
        
        if high_outlier_features:
            print("  • Features with >5% outliers detected:")
            for feature in high_outlier_features:
                print(f"    - {feature}: {outlier_summary[feature]['percentage']:.1f}% outliers")
            print("  • Consider: Winsorization, log transformation, or robust scaling")
            print("  • Evaluate if outliers represent valid extreme values or errors")
        else:
            print("  • No features with excessive outliers (>5%)")
            print("  • Standard scaling should be sufficient for most features")
        
        return outlier_summary
        
    except Exception as e:
        print(f"❌ Error in outlier detection: {str(e)}")
        return {}


