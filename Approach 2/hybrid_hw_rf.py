import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor

class HWRFHybrid:
    """
    A Hybrid Time Series Forecasting Model:
    Stage 1: Fit Holt-Winters Exponential Smoothing to capture the linear trend and seasonality.
    Stage 2: Fit Random Forest Regressor on the residuals of the Holt-Winters model using lag features.
    """
    def __init__(self, lag_size=7, trend='add', seasonal=None, seasonal_periods=None):
        self.lag_size = lag_size
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        
        self.hw_model = None
        self.hw_results = None
        self.rf_model = None
        
        self.train_y = None
        self.hw_fitted = None
        self.train_residuals = None

    def fit(self, y_train):
        self.train_y = np.array(y_train, dtype=float)
        n_samples = len(self.train_y)
        
        if n_samples <= self.lag_size + 1:
            raise ValueError(f"Training data size ({n_samples}) must be greater than lag_size + 1 ({self.lag_size + 1}).")

        print("Stage 1: Fitting Holt-Winters Exponential Smoothing...")
        # Fit Holt-Winters Exponential Smoothing
        # Wrap in try/except to fall back to double smoothing if triple fails to optimize
        try:
            self.hw_model = ExponentialSmoothing(
                self.train_y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
            self.hw_results = self.hw_model.fit()
        except Exception as e:
            print(f"Warning: Holt-Winters with seasonal='{self.seasonal}' failed to fit: {e}")
            print("Falling back to Double Exponential Smoothing (no seasonal component)...")
            self.hw_model = ExponentialSmoothing(
                self.train_y,
                trend=self.trend,
                seasonal=None
            )
            self.hw_results = self.hw_model.fit()

        # Compute fitted values (in-sample predictions)
        self.hw_fitted = self.hw_results.fittedvalues
        
        # Calculate residuals
        self.train_residuals = self.train_y - self.hw_fitted
        
        print("Stage 2: Preparing residuals lag features for Random Forest...")
        X_res, y_res = [], []
        for t in range(self.lag_size, n_samples):
            X_res.append(self.train_residuals[t - self.lag_size : t][::-1])  # Reverse so e_{t-1} is first
            y_res.append(self.train_residuals[t])
            
        X_res = np.array(X_res)
        y_res = np.array(y_res)

        print("Fitting Random Forest on residuals...")
        # Fit Random Forest Regressor
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_res, y_res)
        print("Model training complete.")

    def predict(self, steps):
        if self.hw_results is None or self.rf_model is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
            
        # 1. Forecast the trend using Holt-Winters
        hw_forecast = self.hw_results.forecast(steps)
        
        # 2. Forecast the residuals recursively using Random Forest
        residuals_history = list(self.train_residuals)
        rf_forecast = []
        
        for step in range(steps):
            lags = np.array(residuals_history[-self.lag_size:])[::-1]
            lags = lags.reshape(1, -1)
            
            # Predict next residual
            pred_res = float(self.rf_model.predict(lags)[0])
            rf_forecast.append(pred_res)
            
            # Append predicted residual to history for recursive lags
            residuals_history.append(pred_res)
            
        # 3. Combine forecasts
        hybrid_forecast = hw_forecast + np.array(rf_forecast)
        
        return hybrid_forecast, hw_forecast, np.array(rf_forecast)

    def predict_in_sample(self):
        if self.hw_results is None or self.rf_model is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
            
        n_samples = len(self.train_y)
        hybrid_fitted = np.zeros(n_samples)
        
        hybrid_fitted[:self.lag_size] = self.hw_fitted[:self.lag_size]
        
        X_res = []
        for t in range(self.lag_size, n_samples):
            X_res.append(self.train_residuals[t - self.lag_size : t][::-1])
        X_res = np.array(X_res)
        
        pred_res = self.rf_model.predict(X_res)
        hybrid_fitted[self.lag_size:] = self.hw_fitted[self.lag_size:] + pred_res
        
        return hybrid_fitted
