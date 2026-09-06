import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor

from hybrid_hw_rf import HWRFHybrid

class RFBaseline:
    """
    Baseline model using only Random Forest with lag features of the target series.
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
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
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
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    
    if n - (k + 1) > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - (k + 1))
    else:
        adj_r2 = r2
        
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
    print(f"Loaded processed daily data: {len(df)} days.")
    
    test_size = 10
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    targets = ["Confirmed_Daily", "Deaths_Daily", "Recovered_Daily"]
    results_summary = []
    
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False)
    
    for i, target in enumerate(targets):
        print(f"\n==========================================")
        print(f"Evaluating daily target: {target}")
        print(f"==========================================")
        
        train_y = train_df[target].values
        test_y = test_df[target].values
        
        # --- 1. Fit Baseline Holt-Winters ---
        print("\nFitting Baseline Holt-Winters...")
        try:
            hw_baseline = ExponentialSmoothing(
                train_y,
                trend='add',
                seasonal='add',
                seasonal_periods=7
            )
            hw_res = hw_baseline.fit()
        except Exception as e:
            print(f"Warning: Baseline Holt-Winters seasonal fit failed: {e}. Falling back to double smoothing...")
            hw_baseline = ExponentialSmoothing(
                train_y,
                trend='add',
                seasonal=None
            )
            hw_res = hw_baseline.fit()
        hw_preds = hw_res.forecast(test_size)
        
        # --- 2. Fit Baseline Random Forest ---
        print("Fitting Baseline Random Forest...")
        rf_baseline = RFBaseline(lag_size=7)
        rf_baseline.fit(train_y)
        rf_preds = rf_baseline.predict(test_size)
        
        # --- 3. Fit Hybrid Model ---
        print("Fitting Hybrid Holt-Winters + Random Forest...")
        hybrid_model = HWRFHybrid(lag_size=7, trend='add', seasonal='add', seasonal_periods=7)
        hybrid_model.fit(train_y)
        hybrid_preds, _, _ = hybrid_model.predict(test_size)
        
        # Calculate metrics
        hw_metrics = calculate_metrics(test_y, hw_preds)
        rf_metrics = calculate_metrics(test_y, rf_preds)
        hybrid_metrics = calculate_metrics(test_y, hybrid_preds)
        
        results_summary.append({
            "Target": target,
            "Model": "Baseline Holt-Winters",
            **hw_metrics
        })
        results_summary.append({
            "Target": target,
            "Model": "Baseline Random Forest",
            **rf_metrics
        })
        results_summary.append({
            "Target": target,
            "Model": "Hybrid HW-RF (Ours)",
            **hybrid_metrics
        })
        
        # Plotting
        ax = axes[i]
        dates_train = pd.to_datetime(train_df["Date"].iloc[-20:])
        dates_test = pd.to_datetime(test_df["Date"])
        
        ax.plot(dates_train, train_df[target].iloc[-20:], label="Training Actuals", color="black", linestyle="--")
        ax.plot(dates_test, test_y, label="Test Actuals", color="black", linewidth=2.5)
        
        ax.plot(dates_test, hw_preds, label="Baseline Holt-Winters", marker="o")
        ax.plot(dates_test, rf_preds, label="Baseline Random Forest", marker="s")
        ax.plot(dates_test, hybrid_preds, label="Hybrid HW-RF (Ours)", marker="^", linewidth=2)
        
        ax.set_title(f"Daily Forecast Comparison for {target.replace('_Daily', '')} Cases")
        ax.set_ylabel("Daily Count")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        
    plt.tight_layout()
    plot_filename = "forecast_comparison.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nForecast comparison plot saved as {plot_filename}.")
    
    summary_df = pd.DataFrame(results_summary)
    print("\n\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(summary_df.to_string(index=False))
    
    # Save as Markdown
    markdown_table = summary_df.to_markdown(index=False)
    with open("evaluation_results.md", "w") as f:
        f.write("# Model Evaluation Results Summary (Daily Forecasts)\n\n")
        f.write(markdown_table)
        f.write("\n")
    print("\nSaved evaluation results table as evaluation_results.md.")

if __name__ == "__main__":
    run_evaluation()
