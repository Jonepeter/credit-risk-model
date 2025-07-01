# src/data_processing.py

import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer to engineer time-based and aggregate features
    from raw transaction data, producing a customer-level DataFrame.

    This transformer performs the following steps:
    1. Converts 'TransactionStartTime' to datetime objects.
    2. Extracts time-based features (hour, day, month, year) from transactions.
    3. Aggregates transaction-level data to create customer-level features
       based on 'CustomerId'. These include:
       - Total, average, and standard deviation of transaction 'Amount'.
       - Total and average 'Value'.
       - Count of transactions.
       - Count and rate of fraudulent transactions.
       - Average transaction hour, day, month, and min/max transaction year.
    4. Aggregates categorical features by counting unique values and determining
       the most frequent category (mode) for each customer.
    The output is a DataFrame where each row represents a unique customer,
    with all engineered features.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer. This transformer does not learn any parameters during fit.
        """
        return self

    def transform(self, X):
        """
        Transforms the input DataFrame by engineering new features.

        Args:
            X (pd.DataFrame): The input DataFrame containing raw transaction data.

        Returns:
            pd.DataFrame: A DataFrame with one row per unique CustomerId,
                          containing all engineered features.
        """
        # Ensure X is a DataFrame and create a copy to avoid modifying the original
        X_copy = X.copy()

        # Convert 'TransactionStartTime' to datetime objects for time-based feature extraction
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])

        # Extract time-based features from transaction start time
        X_copy['transaction_hour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['transaction_day'] = X_copy['TransactionStartTime'].dt.day
        X_copy['transaction_month'] = X_copy['TransactionStartTime'].dt.month
        X_copy['transaction_year'] = X_copy['TransactionStartTime'].dt.year
        X_copy['transaction_dayofweek'] = X_copy['TransactionStartTime'].dt.dayofweek # Day of the week (0=Monday, 6=Sunday)

        # Create a base customer DataFrame with unique CustomerIds
        customer_df = pd.DataFrame({'CustomerId': X_copy['CustomerId'].unique()})

        # --- Aggregate Numerical Features per CustomerId ---
        # Group by CustomerId to calculate aggregate statistics for numerical columns
        numerical_agg = X_copy.groupby('CustomerId').agg(
            total_transaction_amount=('Amount', 'sum'),
            avg_transaction_amount=('Amount', 'mean'),
            transaction_count=('TransactionId', 'count'),
            std_transaction_amount=('Amount', 'std'),
            total_value=('Value', 'sum'),
            avg_value=('Value', 'mean'),
            fraud_transaction_count=('FraudResult', 'sum'), # Sum of FraudResult (1s and 0s) gives count of fraudulent transactions
            fraud_rate=('FraudResult', 'mean'), # Mean of FraudResult gives proportion of fraudulent transactions
            ).reset_index()

        # Fill NaN in std_transaction_amount for customers with only one transaction (std dev of single value is NaN)
        numerical_agg['std_transaction_amount'] = numerical_agg['std_transaction_amount'].fillna(0)

        # Merge aggregated numerical features to the base customer DataFrame
        customer_df = pd.merge(customer_df, numerical_agg, on='CustomerId', how='left')

        # --- Aggregate Categorical Features per CustomerId ---
        # Define categorical columns from the original dataset that we want to aggregate
        categorical_cols_to_agg = [
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
            'ChannelId', 'PricingStrategy'
        ]

        for col in categorical_cols_to_agg:
            # Count unique categories for each customer
            unique_counts = X_copy.groupby('CustomerId')[col].nunique().reset_index()
            unique_counts.rename(columns={col: f'unique_{col}_count'}, inplace=True)
            customer_df = pd.merge(customer_df, unique_counts, on='CustomerId', how='left')

            # Get the most frequent category (mode) for each customer
            # Use a lambda function to handle cases where mode might return multiple values (take the first)
            mode_cat = X_copy.groupby('CustomerId')[col].agg(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            ).reset_index()
            mode_cat.rename(columns={col: f'mode_{col}'}, inplace=True)
            customer_df = pd.merge(customer_df, mode_cat, on='CustomerId', how='left')

        return customer_df

def get_full_preprocessing_pipeline():
    """
    Returns a scikit-learn Pipeline for end-to-end data preprocessing.
    This pipeline includes custom feature engineering, handling missing values,
    and encoding/scaling of features.

    The pipeline assumes that the raw input data will have columns like
    'TransactionStartTime', 'Amount', 'Value', 'CustomerId', 'FraudResult',
    and other categorical identifiers.

    Returns:
        sklearn.pipeline.Pipeline: A fitted scikit-learn pipeline object
                                   ready for transforming raw data.
    """
    # Define numerical features that will be present AFTER FeatureEngineer transformation
    numerical_features = [
        'total_transaction_amount', 'avg_transaction_amount', 'transaction_count',
        'std_transaction_amount', 'total_value', 'avg_value',
        'fraud_transaction_count', 'fraud_rate'
    ]
    # Add unique counts for categorical features, which are numerical
    numerical_features.extend([
        f'unique_{col}_count' for col in [
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
            'ChannelId', 'PricingStrategy'
        ]
    ])

    # Define categorical features that will be present AFTER FeatureEngineer transformation
    # These are the 'mode_{category_name}' columns
    categorical_features = [
        f'mode_{col}' for col in [
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
            'ChannelId', 'PricingStrategy'
        ]
    ]

    # Preprocessing for numerical features: impute missing values with median, then standardize
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features: impute missing values with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'handle_unknown' to manage unseen categories in new data
    ])

    # Create a ColumnTransformer to apply different transformations to different subsets of columns
    # The 'remainder="drop"' ensures that any columns not explicitly listed (like 'CustomerId')
    # are dropped from the final feature set passed to the model.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Combine the custom FeatureEngineer and the ColumnTransformer into a single pipeline
    full_pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()), # First step: custom feature engineering
        ('preprocessor', preprocessor)           # Second step: imputation, encoding, scaling
    ])

    return full_pipeline

def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_processed_data(df, file_path):
    """
    Saves a pandas DataFrame to a CSV file. Creates directories if they don't exist.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path where the DataFrame should be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")


    # Example usage and demonstration of the pipeline
    RAW_DATA_PATH = 'data/raw/transactions.csv'
    PROCESSED_DATA_PATH = 'data/processed/processed_features_for_model.csv'
    CUSTOMER_LEVEL_DATA_PATH = 'data/processed/customer_level_raw_features.csv' # For Task 4

    # --- Create dummy data for testing if the file doesn't exist ---
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Creating dummy data at {RAW_DATA_PATH} for demonstration.")
        dummy_data = {
            'TransactionId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'BatchId': [101, 101, 102, 102, 103, 103, 104, 104, 105, 105],
            'AccountId': [1001, 1001, 1002, 1002, 1001, 1003, 1003, 1002, 1004, 1004],
            'SubscriptionId': [201, 201, 202, 202, 201, 203, 203, 202, 204, 204],
            'CustomerId': ['C001', 'C001', 'C002', 'C002', 'C001', 'C003', 'C003', 'C002', 'C004', 'C004'],
            'CurrencyCode': ['UGX', 'UGX', 'KES', 'KES', 'UGX', 'UGX', 'KES', 'UGX', 'UGX', 'KES'],
            'CountryCode': [256, 256, 254, 254, 256, 256, 254, 256, 256, 254],
            'ProviderId': [1, 2, 1, 3, 1, 2, 1, 3, 4, 1],
            'ProductId': ['P01', 'P02', 'P01', 'P03', 'P01', 'P02', 'P01', 'P03', 'P04', 'P01'],
            'ProductCategory': ['CatA', 'CatB', 'CatA', 'CatC', 'CatA', 'CatB', 'CatA', 'CatC', 'CatD', 'CatA'],
            'ChannelId': [1, 2, 1, 3, 1, 2, 1, 3, 4, 1],
            'Amount': [1000, 500, 2000, 1500, 750, 300, 1200, 800, 2500, 600],
            'Value': [1000, 500, 2000, 1500, 750, 300, 1200, 800, 2500, 600],
            'TransactionStartTime': [
                '2023-01-01 10:30:00', '2023-01-01 11:00:00',
                '2023-01-02 14:00:00', '2023-01-02 15:00:00',
                '2023-01-03 09:00:00', '2023-01-03 16:00:00',
                '2023-01-04 10:00:00', '2023-01-04 11:00:00',
                '2023-01-05 12:00:00', '2023-01-05 13:00:00'
            ],
            'PricingStrategy': [1, 2, 1, 3, 1, 2, 1, 3, 4, 1],
            'FraudResult': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        }
        dummy_df = pd.DataFrame(dummy_data)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        dummy_df.to_csv(RAW_DATA_PATH, index=False)

    # Load raw data
    raw_df = load_data(RAW_DATA_PATH)

    if raw_df is not None:
        print("\n--- Step 1: Running FeatureEngineer to get customer-level data ---")
        # Instantiate FeatureEngineer separately to get the customer-level DataFrame
        # This DataFrame will be used for RFM calculation in Task 4 before full preprocessing
        feature_engineer_transformer = FeatureEngineer()
        customer_level_df = feature_engineer_transformer.fit_transform(raw_df)
        print(f"Customer-level data shape: {customer_level_df.shape}")
        print("First 5 rows of customer-level data:")
        print(customer_level_df.head())
        save_processed_data(customer_level_df, CUSTOMER_LEVEL_DATA_PATH)


        print("\n--- Step 2: Running the full preprocessing pipeline ---")
        # Get the full preprocessing pipeline
        full_preprocessing_pipeline = get_full_preprocessing_pipeline()

        # Fit and transform the raw data using the full pipeline
        # The output `processed_data_array` will be a NumPy array
        # containing the scaled numerical features and one-hot encoded categorical features.
        processed_data_array = full_preprocessing_pipeline.fit_transform(raw_df)

        print(f"Shape of fully processed data (NumPy array): {processed_data_array.shape}")

        # To get the column names for the processed data (useful for inspection/debugging)
        # This requires accessing the fitted transformers within the pipeline
        try:
            # Get the preprocessor step from the full pipeline
            preprocessor_step = full_preprocessing_pipeline.named_steps['preprocessor']

            # Get the feature names for numerical features (they remain the same)
            numerical_feature_names = preprocessor_step.transformers_[0][2]

            # Get the feature names for one-hot encoded categorical features
            onehot_encoder = preprocessor_step.named_transformers_['cat'].named_steps['onehot']
            categorical_feature_names = onehot_encoder.get_feature_names_out(
                preprocessor_step.transformers_[1][2] # Pass the original categorical column names
            )

            # Combine all feature names
            all_processed_feature_names = list(numerical_feature_names) + list(categorical_feature_names)

            # Convert the processed NumPy array back to a DataFrame for saving/inspection
            processed_df = pd.DataFrame(processed_data_array, columns=all_processed_feature_names)
            print("\nFirst 5 rows of fully processed (model-ready) data:")
            print(processed_df.head())
            print(f"Fully processed data shape: {processed_df.shape}")

            # Save the fully processed data
            save_processed_data(processed_df, PROCESSED_DATA_PATH)

        except Exception as e:
            print(f"Could not reconstruct DataFrame with column names after full preprocessing: {e}")
            print("The raw NumPy array of processed features is available in 'processed_data_array'.")