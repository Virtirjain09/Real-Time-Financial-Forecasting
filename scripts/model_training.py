import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    return model

def create_hybrid_model(input_shape):
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
    
    rnn = Sequential([
        LSTM(100, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    return cnn, rnn, rf_model

def train_hybrid_model(X_train, y_train, input_shape):
    cnn_model, rnn_model, rf_model = create_hybrid_model(input_shape)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    lr_reducer = ReduceLROnPlateau(factor=0.5, patience=5)
    
    cnn_model.compile(optimizer='adam', loss='mse')
    rnn_model.compile(optimizer='adam', loss='mse')
    
    cnn_model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=100, batch_size=32, validation_split=0.2,
        callbacks=[early_stopping, lr_reducer], verbose=0)
    rnn_model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=100, batch_size=32, validation_split=0.2,
        callbacks=[early_stopping, lr_reducer], verbose=0)
    rf_model.fit(X_train, y_train)
    
    return cnn_model, rnn_model, rf_model

def hybrid_predict(cnn_model, rnn_model, rf_model, X):
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
    
    weights = [0.8, 0.1, 0.1]  # RF, CNN, RNN weights
    
    cnn_pred = cnn_model.predict(X_reshaped, verbose=0).flatten()
    rnn_pred = rnn_model.predict(X_reshaped, verbose=0).flatten()
    rf_pred = rf_model.predict(X)
    
    final_prediction = (weights[0] * rf_pred + 
                        weights[1] * cnn_pred + 
                        weights[2] * rnn_pred)
    
    return final_prediction

