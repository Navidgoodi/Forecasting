"""
Project: Hybrid Stock Price Prediction (ARIMAX + CatBoost)
Author: [Your Name]
Date: February 2026
Description: 
    This project implements a hybrid approach for time-series forecasting.
    It combines the linearity of ARIMAX with the non-linear capabilities of CatBoost
    to model residuals. The strategy uses a Walk-Forward (Rolling Window) validation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Data Loading & Feature Engineering
# ==========================================
def get_stock_data(ticker, start_date, end_date):
    """
    Downloads stock data and generates exogenous features.
    Crucial Step: Shifts exogenous variables by 1 day to prevent look-ahead bias.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate Returns (optional for analysis, not used as target here)
    df['Return'] = df['Close'].pct_change()
    
    # --- Feature Engineering ---
    # 1. Moving Average (MA5)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    
    # 2. Lagged Exogenous Variables (Shifted by 1)
    # We must use T-1 data to predict T. 
    # If we use Volume of today to predict Price of today, it's cheating (Data Leakage).
    df['Exog_Vol'] = df['Volume'].shift(1)
    df['Exog_MA5'] = df['MA5'].shift(1)
    
    # Drop NaN values created by rolling/shifting
    df.dropna(inplace=True)
    
    # Define Target (Y) and Exogenous Features (X)
    data = df[['Close', 'Exog_Vol', 'Exog_MA5']].copy()
    
    print(f"Data prepared. Shape: {data.shape}")
    return data

# ==========================================
# 2. ARIMAX Parameter Selection (AIC)
# ==========================================
def optimize_arimax_params(endog, exog, p_range, d_range, q_range):
    """
    Grid Search to find the best (p, d, q) parameters based on AIC score.
    """
    best_aic = float("inf")
    best_order = (1, 1, 1)
    
    # Simple grid search loop
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(endog, exog=exog, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    print(f"Best ARIMAX Order Found: {best_order} | AIC: {best_aic:.2f}")
    return best_order

# ==========================================
# 3. Hybrid Rolling Window Forecast
# ==========================================
def hybrid_rolling_forecast(data, train_ratio=0.8):
    """
    Performs Walk-Forward Validation (Rolling Window):
    1. Train ARIMAX on history -> Forecast Linear Component
    2. Calculate Residuals (Actual - Linear)
    3. Train CatBoost on Residuals -> Forecast Non-Linear Error
    4. Final Forecast = Linear + Non-Linear
    """
    # Split Data
    n = len(data)
    train_size = int(n * train_ratio)
    
    # Prepare Arrays
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    history_y = list(train_data['Close'])
    history_exog = list(zip(train_data['Exog_Vol'], train_data['Exog_MA5']))
    
    predictions = []
    
    print(f"\nStarting Rolling Forecast on {len(test_data)} test points...")
    print("This may take a while as models are retrained at each step...")

    # Find best parameters once using initial training data (to save time)
    # In a rigorous production setting, you might re-optimize parameters periodically.
    best_order = optimize_arimax_params(
        train_data['Close'], 
        train_data[['Exog_Vol', 'Exog_MA5']], 
        p_range=[1, 2], d_range=[1], q_range=[1, 2]
    )

    # Rolling Loop
    for t in range(len(test_data)):
        # 1. Prepare current history for training
        curr_exog_hist = np.array(history_exog)
        curr_y_hist = np.array(history_y)
        
        # 2. Fit ARIMAX
        arimax_model = ARIMA(curr_y_hist, exog=curr_exog_hist, order=best_order).fit()
        
        # 3. Forecast Linear Component (1 step ahead)
        # Get the exogenous variables for the NEXT time step (from test_data)
        next_exog = test_data.iloc[t][['Exog_Vol', 'Exog_MA5']].values.reshape(1, -1)
        arimax_pred = arimax_model.forecast(steps=1, exog=next_exog)[0]
        
        # 4. Model Residuals with CatBoost
        residuals = curr_y_hist - arimax_model.fittedvalues
        
        cb_model = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.1, verbose=False)
        cb_model.fit(curr_exog_hist, residuals)
        
        # 5. Forecast Non-Linear Error
        cb_resid_pred = cb_model.predict(next_exog)[0]
        
        # 6. Combine
        final_pred = arimax_pred + cb_resid_pred
        predictions.append(final_pred)
        
        # 7. Update History (Walk-Forward)
        true_value = test_data.iloc[t]['Close']
        history_y.append(true_value)
        history_exog.append(tuple(next_exog[0]))
        
        if t % 5 == 0:
            print(f"Step {t}/{len(test_data)} | Pred: {final_pred:.2f} | Actual: {true_value:.2f}")

    # Create Result DataFrame
    results_df = test_data.copy()
    results_df['Hybrid_Pred'] = predictions
    return results_df

# ==========================================
# 4. Evaluation & Visualization
# ==========================================
def evaluate_and_plot(results, ticker):
    # Metrics
    mae = mean_absolute_error(results['Close'], results['Hybrid_Pred'])
    rmse = np.sqrt(mean_squared_error(results['Close'], results['Hybrid_Pred']))
    
    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Trading Logic (Simple Strategy)
    # Signal: If Pred > Previous_Close => BUY
    results['Signal'] = np.where(results['Hybrid_Pred'] > results['Close'].shift(1), 1, -1)
    results['Strategy_Return'] = results['Signal'].shift(1) * results['Close'].pct_change()
    
    cum_returns = (1 + results['Strategy_Return']).cumprod()
    
    # --- Plotting ---
    plt.figure(figsize=(14, 7))
    
    # Plot Actual vs Predicted
    plt.plot(results.index, results['Close'], label='Actual Price', color='black', linewidth=2)
    plt.plot(results.index, results['Hybrid_Pred'], label='Hybrid Forecast (ARIMAX+CatBoost)', 
             color='red', linestyle='--', linewidth=1.5)
    
    plt.title(f"{ticker} - Hybrid Model Prediction vs Actual", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Settings
    TICKER = "AAPL"
    START_DATE = "2024-01-01"
    END_DATE = "2025-02-12" # Adjust as needed
    
    # 1. Get Data
    df_data = get_stock_data(TICKER, START_DATE, END_DATE)
    
    # 2. Run Hybrid Model
    forecast_results = hybrid_rolling_forecast(df_data, train_ratio=0.85)
    
    # 3. Visualize
    evaluate_and_plot(forecast_results, TICKER)
