import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, GlobalMaxPool1D, Dropout, BatchNormalization
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import ta


# Set page config
st.set_page_config(page_title="Advanced Stock Market Dashboard", layout="wide")

@st.cache_resource
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            return None, None
        return stock, data
    except Exception as e:
        return None, None

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

def prepare_features(df):
    # Add more sophisticated features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df = add_technical_indicators(df)
    
    # Add more technical indicators
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
    
    # Create more sophisticated lagged features
    for i in range(1, 8):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        df[f'Returns_Lag_{i}'] = df['Returns'].shift(i)
    
    return df


def train_model(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def create_hybrid_model(input_shape):
    # Enhanced CNN
    cnn = Sequential([
        Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Enhanced LSTM
    rnn = Sequential([
        LSTM(100, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Enhanced Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    return cnn, rnn, rf_model


def hybrid_predict(cnn_model, rnn_model, rf_model, X):
    try:
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Adjust prediction weights
        weights = [0.8, 0.1, 0.1]  # RF, CNN, RNN weights
        
        # Ensure proper model compilation
        cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        rnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        cnn_pred = cnn_model.predict(X_reshaped, verbose=0).flatten()
        rnn_pred = rnn_model.predict(X_reshaped, verbose=0).flatten()
        rf_pred = rf_model.predict(X)
        
        # Weighted ensemble prediction
        final_prediction = (weights[0] * rf_pred + 
                          weights[1] * cnn_pred + 
                          weights[2] * rnn_pred)
        
        return final_prediction
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    input_shape = (X_train.shape[1], 1)
    cnn_model, rnn_model, rf_model = create_hybrid_model(input_shape)

    # Add early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    lr_reducer = ReduceLROnPlateau(factor=0.5, patience=5)
    
    cnn_model.compile(optimizer='adam', loss='mse')
    rnn_model.compile(optimizer='adam', loss='mse')
    
    cnn_model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=100, batch_size=32,  validation_split=0.2,
        callbacks=[early_stopping, lr_reducer], verbose=0)
    rnn_model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=100, batch_size=32,  validation_split=0.2,
        callbacks=[early_stopping, lr_reducer], verbose=0)
    rf_model.fit(X_train, y_train)
    
    hybrid_pred = hybrid_predict(cnn_model, rnn_model, rf_model, X_test)
    
    return xgb_model, (cnn_model, rnn_model, rf_model), xgb_pred, hybrid_pred

def calculate_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R² Score": r2_score(y_true, y_pred)
    }

def add_time_controls():
    col1, col2 = st.columns(2)
    with col1:
        time_preset = st.selectbox(
            "Preset Ranges",
            ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y", "MAX"]
        )
    with col2:
        refresh_rate = st.slider(
            "Auto-refresh interval (seconds)",
            5, 60, 30
        )
    return time_preset, refresh_rate

def add_time_controls():
    col1, col2 = st.columns(2)
    with col1:
        time_preset = st.selectbox(
            "Preset Ranges",
            ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y", "MAX"]
        )
    with col2:
        refresh_rate = st.slider(
            "Auto-refresh interval (seconds)",
            5, 60, 30
        )
    return time_preset, refresh_rate

def create_interactive_chart(fig, df):
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )
    )
    
    # Add range selector
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        template='plotly_dark',
        height=600
    )
    
    return fig


def add_indicator_controls():
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators",
        ["MA20", "MA50", "RSI", "MACD", "Bollinger Bands"],
        default=["MA20", "MA50"]
    )
    
    if "RSI" in indicators:
        rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14)
    if "MACD" in indicators:
        macd_fast = st.sidebar.slider("MACD Fast Period", 8, 20, 12)
        macd_slow = st.sidebar.slider("MACD Slow Period", 20, 40, 26)
    
    return indicators

def add_volume_analysis(fig, df):
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            yaxis='y2'
        )
    )
    
    fig.update_layout(
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right"
        )
    )
    return fig

def prepare_data(X_scaled, y):
    if X_scaled is None or y is None:
        return None, None
        
    try:
        # Ensure proper data shapes
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        return X_reshaped, y
    except Exception as e:
        print(f"Data preparation failed: {str(e)}")
        return None, None
 


def main():
    st.title("Stock Market Dashboard")
    st.write("Analyze stocks with technical indicators and forecasting using  XGBoost and Hybrid Model Metrics")

    ticker = st.sidebar.text_input("Enter Stock Ticker", "")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-20"))
    forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)

    time_preset, refresh_rate = add_time_controls()
    indicators = add_indicator_controls()
    # alert_price, alert_condition, alert_active = add_price_alerts()
    fig = go.Figure()
    
    # Get stock data
    stock, data = get_stock_data(ticker, start_date, end_date)
    
    if stock is not None and not data.empty:
        df = prepare_features(data.copy())
        df.dropna(inplace=True)
        fig = create_interactive_chart(fig, df)
        if "Volume" in indicators:
            fig = add_volume_analysis(fig, df)
            
        # Update chart based on selected indicators
        for indicator in indicators:
            if indicator in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[indicator],
                    name=indicator
                ))
   
    
    if stock is not None and not data.empty:
        try:
            df = prepare_features(data.copy())
            df.dropna(inplace=True)

            st.header("Company Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Company Name:** {stock.info['longName']}")
                st.write(f"**Sector:** {stock.info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {stock.info.get('industry', 'N/A')}")
            with col2:
                st.write(f"**Market Cap:** ${stock.info.get('marketCap', 'N/A'):,}")
                st.write(f"**P/E Ratio:** {stock.info.get('trailingPE', 'N/A'):.2f}")
                st.write(f"**Dividend Yield:** {stock.info.get('dividendYield', 'N/A'):.2%}")

            st.subheader("Stock Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20-day MA'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-day MA'))
            fig.update_layout(template='plotly_dark', height=600)
            st.plotly_chart(fig, use_container_width=True)

            st.header("Price Forecasting")
            
            feature_columns = ['Volume', 'Returns', 'Volatility', 'RSI', 'MACD', 'MA20', 'MA50', 'MACD_Signal']
            X = df[feature_columns].values
            y = df['Close'].values
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            xgb_model = train_model(X_scaled[:-forecast_days], y[:-forecast_days])
            xgb_predictions = xgb_model.predict(X_scaled[-forecast_days:])

            try:
                input_shape = (X_scaled.shape[1], 1)
                cnn_model, rnn_model, rf_model = create_hybrid_model(input_shape)
                rf_model.fit(X_scaled[:-forecast_days], y[:-forecast_days])
                hybrid_predictions = hybrid_predict(cnn_model, rnn_model, rf_model, X_scaled[-forecast_days:])
            except Exception as e:
                st.error(f"Error in hybrid model creation: {str(e)}")
                hybrid_predictions = np.zeros_like(xgb_predictions)

            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
            xgb_model, hybrid_models, xgb_pred, hybrid_pred = train_and_evaluate_models(X_train, y_train, X_test, y_test)
            
            forecast_df = pd.DataFrame({
                'Date': pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days),
                'XGBoost_Prediction': xgb_predictions,
                'Hybrid_Prediction': hybrid_predictions
            }).set_index('Date')

            st.subheader(f"Model Comparison - Price Predictions for Next {forecast_days} Days")
            st.write(forecast_df)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical Price'))
            fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['XGBoost_Prediction'], name='XGBoost Forecast', line=dict(dash='dash')))
            fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Hybrid_Prediction'], name='Hybrid Forecast', line=dict(dash='dot')))
            fig2.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### XGBoost Metrics")
                metrics = calculate_metrics(y_test, xgb_pred)
                for name, value in metrics.items():
                    st.metric(name, f"{value:.2f}")
            with col2:
                st.markdown("### Hybrid Model Metrics")
                metrics = calculate_metrics(y_test, hybrid_pred)
                for name, value in metrics.items():
                    st.metric(name, f"{value:.2f}")
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.error("Error: Invalid ticker symbol or no data available")

if __name__ == "__main__":
    main()
