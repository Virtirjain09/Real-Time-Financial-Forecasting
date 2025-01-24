import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data_for_modeling(df):
    feature_columns = ['Volume', 'Returns', 'Volatility', 'RSI', 'MACD', 'MA20', 'MA50', 'MACD_Signal']
    X = df[feature_columns].values
    y = df['Close'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def calculate_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def calculate_volatility(returns, window=20):
    return returns.rolling(window=window).std().dropna()

