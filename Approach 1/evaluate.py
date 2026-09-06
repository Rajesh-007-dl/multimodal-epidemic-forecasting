import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pmdarima.arima import auto_arima
from xgboost import XGBRegressor

from hybrid_arima_xgboost import ARIMAXGBoostHybrid

class XGBoostBaseline:
    """
    Baseline model using only XGBoost with lag features of the target series.
    """
    def __init__(self, lag_size=7):
        self.lag_size = lag_size
        self.model = None
        self.train_y = None

    def fit(self, y_train):
        self.train_y = np.array(y_train, dtype=float)
        n_samples = len(self.train_y)
        
        X, y = [], []
        for t in range(self.lag_size, n_samples):
            X.append(self.train_y[t - self.lag_size : t][::-1])
            y.append(self.train_y[t])
            
        X = np.array(X)
        y = np.array(y)
        
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

    def predict(self, steps):
        history = list(self.train_y)
        predictions = []
        
        for step in range(steps):
            lags = np.array(history[-self.lag_size:])[::-1]
            lags = lags.reshape(1, -1)
            pred = float(self.model.predict(lags)[0])
            predictions.append(pred)
            history.append(pred)
            
        return np.array(predictions)

def calculate_metrics(y_true, y_pred, k=1):
    """
    Calculates R2, Adjusted R2, MSE, MAE, and RMSE.
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    
    # Formula for adjusted R2: 1 - (1 - R2) * (n - 1) / (n - (k + 1))
    if n - (k + 1) > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - (k + 1))
    else:
        adj_r2 = r2  # Fallback if degrees of freedom are too low
        
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        "R2": r2,
        "Adj_R2": adj_r2,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse
    }

def run_evaluation():
    data_path = "global_covid_daily.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run preprocess.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded processed data: {len(df)} days.")
    
    # Validation scheme: Train on all but final 10 days, test on final 10 days
    test_size = 10
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    targets = ["Confirmed_Daily", "Deaths_Daily", "Recovered_Daily"]
    results_summary = []
    
    # Set up plotting
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False)
    
    for i, target in enumerate(targets):
        print(f"\n==========================================")
        print(f"Evaluating target: {target}")
        print(f"==========================================")
        
        train_y = train_df[target].values
        test_y = test_df[target].values
        
        # --- 1. Fit Baseline Auto-ARIMA ---
        print("\nFitting Baseline Auto-ARIMA...")
        arima_baseline = auto_arima(
            train_y,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            seasonal=True, m=7,
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )
        arima_preds = arima_baseline.predict(n_periods=test_size)
        
        # --- 2. Fit Baseline XGBoost ---
        print("Fitting Baseline XGBoost...")
        xgb_baseline = XGBoostBaseline(lag_size=7)
        xgb_baseline.fit(train_y)
        xgb_preds = xgb_baseline.predict(test_size)
        
        # --- 3. Fit Hybrid Model ---
        print("Fitting Hybrid ARIMA-XGBoost...")
        hybrid_model = ARIMAXGBoostHybrid(lag_size=7, seasonal=True, m=7)
        hybrid_model.fit(train_y)
        hybrid_preds, _, _ = hybrid_model.predict(test_size)
        
        # Save metrics
        arima_metrics = calculate_metrics(test_y, arima_preds)
        xgb_metrics = calculate_metrics(test_y, xgb_preds)
        hybrid_metrics = calculate_metrics(test_y, hybrid_preds)
        
        results_summary.append({
            "Target": target,
            "Model": "Baseline Auto-ARIMA",
            **arima_metrics
        })
        results_summary.append({
            "Target": target,
            "Model": "Baseline XGBoost",
            **xgb_metrics
        })
        results_summary.append({
            "Target": target,
            "Model": "Hybrid ARIMA-XGBoost",
            **hybrid_metrics
        })
        
        # Plotting the results
        ax = axes[i]
        dates_train = pd.to_datetime(train_df["Date"].iloc[-20:])  # Show last 20 days of training
        dates_test = pd.to_datetime(test_df["Date"])
        
        # Plot train actuals (last 20 days)
        ax.plot(dates_train, train_df[target].iloc[-20:], label="Training Actuals", color="black", linestyle="--")
        
        # Plot test actuals
        ax.plot(dates_test, test_y, label="Test Actuals", color="black", linewidth=2.5)
        
        # Plot model predictions
        ax.plot(dates_test, arima_preds, label="Baseline Auto-ARIMA", marker="o")
        ax.plot(dates_test, xgb_preds, label="Baseline XGBoost", marker="s")
        ax.plot(dates_test, hybrid_preds, label="Hybrid ARIMA-XGBoost", marker="^", linewidth=2)
        
        ax.set_title(f"Forecast Comparison for {target.replace('_Daily', '')} Cases")
        ax.set_ylabel("Daily Count")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        
    plt.tight_layout()
    plot_filename = "forecast_comparison.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nForecast comparison plot saved as {plot_filename}.")
    
    # Print results summary table
    summary_df = pd.DataFrame(results_summary)
    print("\n\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    
    # Format table for output
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(summary_df.to_string(index=False))
    
    # Also save summary table as Markdown file
    markdown_table = summary_df.to_markdown(index=False)
    with open("evaluation_results.md", "w") as f:
        f.write("# Model Evaluation Results Summary\n\n")
        f.write(markdown_table)
        f.write("\n")
    print("\nSaved evaluation results table as evaluation_results.md.")

if __name__ == "__main__":
    run_evaluation()
