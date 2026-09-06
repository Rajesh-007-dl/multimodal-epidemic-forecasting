import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from xgboost import XGBRegressor

class ARIMAXGBoostHybrid:
    """
    A Hybrid Time Series Forecasting Model:
    Stage 1: Fit Auto-ARIMA to capture the linear trend and seasonality.
    Stage 2: Fit XGBoost on the residuals (errors) of the ARIMA model using lag features.
    """
    def __init__(self, lag_size=7, seasonal=True, m=7):
        self.lag_size = lag_size
        self.seasonal = seasonal
        self.m = m
        self.arima_model = None
        self.xgboost_model = None
        
        # Keep track of training values and residuals
        self.train_y = None
        self.arima_fitted = None
        self.train_residuals = None

    def fit(self, y_train):
        """
        Fits the ARIMA model to the training data, computes residuals,
        and trains XGBoost on those residuals using lagged features.
        """
        self.train_y = np.array(y_train, dtype=float)
        n_samples = len(self.train_y)
        
        if n_samples <= self.lag_size + 1:
            raise ValueError(f"Training data size ({n_samples}) must be greater than lag_size + 1 ({self.lag_size + 1}).")

        print("Stage 1: Fitting Auto-ARIMA...")
        # Fit Auto-ARIMA
        self.arima_model = auto_arima(
            self.train_y,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            m=self.m if self.seasonal else 1,
            seasonal=self.seasonal,
            d=None, D=None,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        print(f"Auto-ARIMA selected order: {self.arima_model.order} x {self.arima_model.seasonal_order}")

        # Compute fitted values (in-sample predictions)
        self.arima_fitted = self.arima_model.predict_in_sample()
        
        # Calculate residuals
        self.train_residuals = self.train_y - self.arima_fitted
        
        print("Stage 2: Preparing residuals lag features for XGBoost...")
        # Prepare lag features for XGBoost
        # Features X: lags of residuals [e_{t-1}, e_{t-2}, ..., e_{t-L}]
        # Target y: current residual e_t
        X_res, y_res = [], []
        for t in range(self.lag_size, n_samples):
            X_res.append(self.train_residuals[t - self.lag_size : t][::-1])  # Reverse so e_{t-1} is first
            y_res.append(self.train_residuals[t])
            
        X_res = np.array(X_res)
        y_res = np.array(y_res)

        print("Fitting XGBoost on residuals...")
        # Fit XGBoost Regressor
        self.xgboost_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        )
        self.xgboost_model.fit(X_res, y_res)
        print("Model training complete.")

    def predict(self, steps):
        """
        Forecasts out-of-sample values recursively for the next 'steps' periods.
        """
        if self.arima_model is None or self.xgboost_model is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
            
        # 1. Forecast the linear part using ARIMA
        arima_forecast = self.arima_model.predict(n_periods=steps)
        
        # 2. Forecast the residuals recursively using XGBoost
        # Start with the end of our training residuals
        residuals_history = list(self.train_residuals)
        xgboost_forecast = []
        
        for step in range(steps):
            # Take the last lag_size residuals to construct the feature vector
            lags = np.array(residuals_history[-self.lag_size:])[::-1]  # Reverse to match training order [e_{t-1}, e_{t-2}, ...]
            lags = lags.reshape(1, -1)
            
            # Predict next residual
            pred_res = float(self.xgboost_model.predict(lags)[0])
            xgboost_forecast.append(pred_res)
            
            # Append predicted residual to history for the next step's lag features
            residuals_history.append(pred_res)
            
        # 3. Combine both forecasts
        hybrid_forecast = arima_forecast + np.array(xgboost_forecast)
        
        return hybrid_forecast, arima_forecast, np.array(xgboost_forecast)

    def predict_in_sample(self):
        """
        Generates in-sample fitted values on the training set.
        """
        if self.arima_model is None or self.xgboost_model is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
            
        n_samples = len(self.train_y)
        hybrid_fitted = np.zeros(n_samples)
        
        # For dates t < lag_size, we don't have enough lags for XGBoost, so use raw ARIMA
        hybrid_fitted[:self.lag_size] = self.arima_fitted[:self.lag_size]
        
        # For dates t >= lag_size, use ARIMA prediction + XGBoost prediction of residual
        X_res = []
        for t in range(self.lag_size, n_samples):
            X_res.append(self.train_residuals[t - self.lag_size : t][::-1])
        X_res = np.array(X_res)
        
        pred_res = self.xgboost_model.predict(X_res)
        
        hybrid_fitted[self.lag_size:] = self.arima_fitted[self.lag_size:] + pred_res
        
        return hybrid_fitted
