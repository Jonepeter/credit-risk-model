import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from datetime import datetime

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer time-based and aggregate features from raw transaction data.
    Produces a customer-level DataFrame with engineered features.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit method. Does not learn parameters.
        Args:
            X (pd.DataFrame): Input data.
            y: Not used.
        Returns:
            self
        """
        try:
            return self
        except Exception as e:
            print(f"Error in FeatureEngineer.fit: {e}")
            return self

    def transform(self, X):
        """
        Transforms input DataFrame by engineering new features.
        Args:
            X (pd.DataFrame): Raw transaction data.
        Returns:
            pd.DataFrame: Customer-level engineered features.
        """
        try:
            X_copy = X.copy()
            X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
            X_copy['transaction_hour'] = X_copy['TransactionStartTime'].dt.hour
            X_copy['transaction_day'] = X_copy['TransactionStartTime'].dt.day
            X_copy['transaction_month'] = X_copy['TransactionStartTime'].dt.month
            X_copy['transaction_year'] = X_copy['TransactionStartTime'].dt.year
            X_copy['transaction_dayofweek'] = X_copy['TransactionStartTime'].dt.dayofweek
            customer_df = pd.DataFrame({'CustomerId': X_copy['CustomerId'].unique()})
            numerical_agg = X_copy.groupby('CustomerId').agg(
                total_transaction_amount=('Amount', 'sum'),
                avg_transaction_amount=('Amount', 'mean'),
                transaction_count=('TransactionId', 'count'),
                std_transaction_amount=('Amount', 'std'),
                total_value=('Value', 'sum'),
                avg_value=('Value', 'mean'),
                fraud_transaction_count=('FraudResult', 'sum'),
                fraud_rate=('FraudResult', 'mean'),
                avg_transaction_hour=('transaction_hour', 'mean'),
                avg_transaction_day=('transaction_day', 'mean'),
                avg_transaction_month=('transaction_month', 'mean'),
                min_transaction_year=('transaction_year', 'min'),
                max_transaction_year=('transaction_year', 'max'),
                avg_transaction_dayofweek=('transaction_dayofweek', 'mean'),
                last_transaction_date=('TransactionStartTime', 'max')
            ).reset_index()
            numerical_agg['std_transaction_amount'] = numerical_agg['std_transaction_amount'].fillna(0)
            customer_df = pd.merge(customer_df, numerical_agg, on='CustomerId', how='left')
            categorical_cols_to_agg = [
                'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
                'ChannelId', 'PricingStrategy'
            ]
            for col in categorical_cols_to_agg:
                unique_counts = X_copy.groupby('CustomerId')[col].nunique().reset_index()
                unique_counts.rename(columns={col: f'unique_{col}_count'}, inplace=True)
                customer_df = pd.merge(customer_df, unique_counts, on='CustomerId', how='left')
                mode_cat = X_copy.groupby('CustomerId')[col].agg(
                    lambda x: x.mode()[0] if not x.mode().empty else np.nan
                ).reset_index()
                mode_cat.rename(columns={col: f'mode_{col}'}, inplace=True)
                customer_df = pd.merge(customer_df, mode_cat, on='CustomerId', how='left')
            return customer_df
        except Exception as e:
            print(f"Error in FeatureEngineer.transform: {e}")
            return None

class RFMClusterer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate RFM metrics, cluster customers, and assign a 'is_high_risk' label.
    """
    def __init__(self, snapshot_date=None, n_clusters=3, random_state=42):
        self.snapshot_date = pd.to_datetime(snapshot_date) if snapshot_date else datetime.now()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.rfm_scaler = None
        self.high_risk_cluster_label = None
        self.rfm_features = ['Recency', 'Frequency', 'Monetary']

    def _calculate_rfm(self, X):
        """
        Calculates Recency, Frequency, and Monetary (RFM) values for each customer.
        Args:
            X (pd.DataFrame): Customer-level data.
        Returns:
            pd.DataFrame: RFM features.
        """
        try:
            rfm_df = pd.DataFrame({'CustomerId': X['CustomerId']})
            rfm_df['Recency'] = (self.snapshot_date - X['last_transaction_date']).dt.days
            rfm_df['Recency'] = rfm_df['Recency'].apply(lambda x: max(0, x))
            rfm_df['Frequency'] = X['transaction_count']
            rfm_df['Monetary'] = X['total_transaction_amount']
            rfm_df['Recency'] = rfm_df['Recency'].fillna(rfm_df['Recency'].max() + 1)
            rfm_df['Frequency'] = rfm_df['Frequency'].fillna(0)
            rfm_df['Monetary'] = rfm_df['Monetary'].fillna(0)
            return rfm_df
        except Exception as e:
            print(f"Error in RFMClusterer._calculate_rfm: {e}")
            return None

    def fit(self, X, y=None):
        """
        Fits KMeans on RFM features and identifies the high-risk cluster.
        Args:
            X (pd.DataFrame): Customer-level features.
            y: Not used.
        Returns:
            self
        """
        try:
            rfm_data = self._calculate_rfm(X)
            rfm_for_clustering = rfm_data[self.rfm_features]
            self.rfm_scaler = StandardScaler()
            scaled_rfm = self.rfm_scaler.fit_transform(rfm_for_clustering)
            self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = self.kmeans_model.fit_predict(scaled_rfm)
            rfm_data['Cluster'] = cluster_labels
            cluster_centroids = pd.DataFrame(self.kmeans_model.cluster_centers_, columns=self.rfm_features)
            cluster_centroids['Cluster'] = cluster_centroids.index
            cluster_centroids['risk_score'] = (
                cluster_centroids['Recency'] - cluster_centroids['Frequency'] - cluster_centroids['Monetary']
            )
            self.high_risk_cluster_label = cluster_centroids.sort_values(by='risk_score', ascending=False).iloc[0]['Cluster']
            print(f"Identified high-risk cluster label: {int(self.high_risk_cluster_label)}")
            print("Cluster Centroids (scaled RFM values):")
            print(cluster_centroids)
            return self
        except Exception as e:
            print(f"Error in RFMClusterer.fit: {e}")
            return self

    def transform(self, X):
        """
        Assigns high-risk label to customers based on RFM clustering.
        Args:
            X (pd.DataFrame): Customer-level features.
        Returns:
            pd.DataFrame: Data with 'is_high_risk' label.
        """
        try:
            if self.kmeans_model is None or self.rfm_scaler is None or self.high_risk_cluster_label is None:
                raise RuntimeError("RFMClusterer has not been fitted yet. Call .fit() first.")
            rfm_data = self._calculate_rfm(X)
            rfm_for_clustering = rfm_data[self.rfm_features]
            scaled_rfm = self.rfm_scaler.transform(rfm_for_clustering)
            cluster_labels = self.kmeans_model.predict(scaled_rfm)
            rfm_data['Cluster'] = cluster_labels
            rfm_data['is_high_risk'] = (rfm_data['Cluster'] == self.high_risk_cluster_label).astype(int)
            X_transformed = pd.merge(X, rfm_data[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
            if 'last_transaction_date' in X_transformed.columns:
                X_transformed = X_transformed.drop(columns=['last_transaction_date'])
            return X_transformed
        except Exception as e:
            print(f"Error in RFMClusterer.transform: {e}")
            return None

def get_full_preprocessing_pipeline(snapshot_date=None):
    """
    Returns a scikit-learn Pipeline for end-to-end data preprocessing, including feature engineering, RFM clustering, and preprocessing.
    Args:
        snapshot_date (str or datetime, optional): Date for Recency calculation.
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline.
    """
    try:
        numerical_features = [
            'total_transaction_amount', 'avg_transaction_amount', 'transaction_count',
            'std_transaction_amount', 'total_value', 'avg_value',
            'fraud_transaction_count', 'fraud_rate',
            'avg_transaction_hour', 'avg_transaction_day', 'avg_transaction_month',
            'min_transaction_year', 'max_transaction_year', 'avg_transaction_dayofweek'
        ]
        numerical_features.extend([
            f'unique_{col}_count' for col in [
                'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
                'ChannelId', 'PricingStrategy'
            ]
        ])
        categorical_features = [
            f'mode_{col}' for col in [
                'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
                'ChannelId', 'PricingStrategy'
            ]
        ]
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        full_pipeline = Pipeline(steps=[
            ('feature_engineer', FeatureEngineer()),
            ('rfm_clusterer', RFMClusterer(snapshot_date=snapshot_date)),
            ('preprocessor', preprocessor)
        ])
        return full_pipeline
    except Exception as e:
        print(f"Error in get_full_preprocessing_pipeline: {e}")
        return None

def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame or None if error.
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
    Saves a pandas DataFrame to a CSV file. Creates directories if needed.
    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the DataFrame.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

def main():
    
    # Load raw data
    raw_df = load_data('../data/raw/data.csv')
    raw_df['TransactionStartTime'] = pd.to_datetime(raw_df['TransactionStartTime']).dt.strftime("%Y-%m-%d %H:%M:%S")
    if raw_df is not None:
        print("\n--- Running the full preprocessing pipeline including RFM and High-Risk Labeling ---")
        # Define a snapshot date for RFM calculation. Use a date after the latest transaction in dummy data.
        # Latest transaction in dummy data is 2024-06-29. Let's use 2024-07-01.
        snapshot_date_for_rfm = '2024-07-01'

        # Get the full preprocessing pipeline
        full_preprocessing_pipeline = get_full_preprocessing_pipeline(snapshot_date=snapshot_date_for_rfm)

        # Fit and transform the raw data using the full pipeline
        # The output `processed_data_array` will be a NumPy array
        # containing the scaled numerical features and one-hot encoded categorical features.
        # The 'is_high_risk' column will be in the DataFrame returned by rfm_clusterer,
        # but will be dropped by the preprocessor as it's the target variable.
        # So, we need to get the 'is_high_risk' column separately if we want to save it with features.

        # To get the target variable 'is_high_risk' along with features,
        # we need to run the pipeline in steps or ensure 'is_high_risk'
        # is preserved. For simplicity in this demonstration, we'll run
        # FeatureEngineer and RFMClusterer first to get the labeled data,
        # then apply the preprocessor.

        # Step 1: Feature Engineering
        feature_engineer_transformer = FeatureEngineer()
        customer_level_df = feature_engineer_transformer.fit_transform(raw_df)
        print(f"Customer-level data shape after FeatureEngineer: {customer_level_df.shape}")
        # print("First 5 rows of customer-level data after FeatureEngineer:")
        # print(customer_level_df.head())
        save_processed_data(customer_level_df, '../data/processed/customer_data.csv')


        # Step 2: RFM Clustering and High-Risk Labeling
        rfm_clusterer_transformer = RFMClusterer(snapshot_date=snapshot_date_for_rfm)
        # Fit and transform to identify clusters and assign labels
        customer_level_with_risk = rfm_clusterer_transformer.fit_transform(customer_level_df)
        print(f"Customer-level data shape after RFMClusterer: {customer_level_with_risk.shape}")
        print("First 5 rows of customer-level data with 'is_high_risk' label:")
        print(customer_level_with_risk.head())

        # Separate features (X) and target (y) before final preprocessing
        # Drop 'CustomerId' as it's not a feature for the model
        X_data_for_preprocessing = customer_level_with_risk.drop(columns=['CustomerId', 'is_high_risk'])
        y_target = customer_level_with_risk['is_high_risk']

        # Step 3: Final Preprocessing (Imputation, Encoding, Scaling)
        # Re-initialize the preprocessor part of the pipeline to apply it to X_data_for_preprocessing
        # We need to ensure the preprocessor only acts on the desired features.
        # The `get_full_preprocessing_pipeline` creates a pipeline that includes FeatureEngineer and RFMClusterer.
        # To just get the preprocessor, we can extract it or define it separately.
        # Let's define it directly for clarity.

        # Define numerical features that will be present in X_data_for_preprocessing
        numerical_features_final = [
            'total_transaction_amount', 'avg_transaction_amount', 'transaction_count',
            'std_transaction_amount', 'total_value', 'avg_value',
            'fraud_transaction_count', 'fraud_rate',
            'avg_transaction_hour', 'avg_transaction_day', 'avg_transaction_month',
            'min_transaction_year', 'max_transaction_year', 'avg_transaction_dayofweek'
        ]
        numerical_features_final.extend([
            f'unique_{col}_count' for col in [
                'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
                'ChannelId', 'PricingStrategy'
            ]
        ])

        # Define categorical features that will be present in X_data_for_preprocessing
        categorical_features_final = [
            f'mode_{col}' for col in [
                'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
                'ChannelId', 'PricingStrategy'
            ]
        ]

        final_preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features_final),
                ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features_final)
            ],
            remainder='drop'
        )

        processed_features_array = final_preprocessor.fit_transform(X_data_for_preprocessing)

        print(f"Shape of fully processed features (NumPy array): {processed_features_array.shape}")
        print(f"Shape of target variable (is_high_risk): {y_target.shape}")

        try:
            # Get feature names for the final processed DataFrame
            numerical_feature_names = final_preprocessor.transformers_[0][2]
            onehot_encoder = final_preprocessor.named_transformers_['cat'].named_steps['onehot']
            categorical_feature_names = onehot_encoder.get_feature_names_out(
                final_preprocessor.transformers_[1][2]
            )
            all_processed_feature_names = list(numerical_feature_names) + list(categorical_feature_names)

            processed_features_df = pd.DataFrame(processed_features_array, columns=all_processed_feature_names)
            processed_features_df['is_high_risk'] = y_target.values # Add target back for saving
            processed_features_df['CustomerId'] = customer_level_with_risk['CustomerId'].values # Add CustomerId back for reference

            print("\nFirst 5 rows of fully processed (model-ready) data with target:")
            print(processed_features_df.head())
            print(f"Fully processed data shape (with target): {processed_features_df.shape}")

            save_processed_data(processed_features_df, '../data/processed/Preprocessed_data.csv')

        except Exception as e:
            print(f"Could not reconstruct DataFrame with column names after full preprocessing: {e}")
            print("The raw NumPy array of processed features is available in 'processed_features_array'.")
            print("The target variable 'is_high_risk' is available in 'y_target'.")