# tests/test_data_processing.py
"""
Unit tests for data_processing.py
"""

import pandas as pd
import numpy as np
import pytest
from src.data_processing import FeatureEngineer, RFMClusterer, woe_iv_select_transform, get_full_preprocessing_pipeline

@pytest.fixture
def sample_transactions():
    # Minimal sample data for testing
    return pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionId': [101, 102, 201, 202, 301],
        'TransactionStartTime': [
            '2018-11-15T02:18:49Z', '2018-11-16T03:20:00Z',
            '2018-11-15T04:00:00Z', '2018-11-17T05:00:00Z',
            '2018-11-18T06:00:00Z'
        ],
        'Amount': [100, 150, 200, 250, 300],
        'Value': [10, 15, 20, 25, 30],
        'FraudResult': [0, 1, 0, 0, 1],
        'CurrencyCode': ['USD', 'USD', 'EUR', 'EUR', 'USD'],
        'CountryCode': ['US', 'US', 'FR', 'FR', 'US'],
        'ProviderId': ['A', 'A', 'B', 'B', 'A'],
        'ProductCategory': ['cat1', 'cat1', 'cat2', 'cat2', 'cat1'],
        'ChannelId': ['web', 'web', 'app', 'app', 'web'],
        'PricingStrategy': ['low', 'low', 'high', 'high', 'low']
    })

def test_feature_engineer(sample_transactions):
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_transactions)
    assert isinstance(result, pd.DataFrame)
    assert 'CustomerId' in result.columns
    assert 'total_transaction_amount' in result.columns

def test_rfm_clusterer(sample_transactions):
    fe = FeatureEngineer()
    customer_df = fe.fit_transform(sample_transactions)
    rfm = RFMClusterer(snapshot_date='2018-11-20')
    result = rfm.fit_transform(customer_df)
    assert 'is_high_risk' in result.columns
    assert set(result['is_high_risk'].unique()).issubset({0, 1})

def test_woe_iv_select_transform(sample_transactions):
    fe = FeatureEngineer()
    customer_df = fe.fit_transform(sample_transactions)
    # Add a fake binary target for testing
    customer_df['is_high_risk'] = [0, 1, 0]
    result = woe_iv_select_transform(customer_df, target_col='is_high_risk')
    assert isinstance(result, pd.DataFrame)
    assert any(col.endswith('_woe') for col in result.columns)

def test_full_pipeline(sample_transactions):
    pipeline = get_full_preprocessing_pipeline(snapshot_date='2018-11-20')
    # Add a fake binary target for testing
    fe = FeatureEngineer()
    customer_df = fe.fit_transform(sample_transactions)
    customer_df['is_high_risk'] = [0, 1, 0]
    # The pipeline expects raw data, so use sample_transactions
    output = pipeline.fit_transform(sample_transactions)
    assert output is not None

def test_placeholder():
    assert True 