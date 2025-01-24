import pandas as pd
import numpy as np
import ta

def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            return None, None
        return stock, data
    except Exception as e:
        return None, None

def prepare_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df = add_technical_indicators(df)
    
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
    
    for i in range(1, 8):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        df[f'Returns_Lag_{i}'] = df['Returns'].shift(i)
    
    return df

def add_technical_indicators(df):
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'])
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    return df

