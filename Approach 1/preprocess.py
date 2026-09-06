import argparse
import pandas as pd
import numpy as np

def preprocess_data(mode="replication"):
    # File paths
    confirmed_path = "time_series_covid19_confirmed_global.csv"
    deaths_path = "time_series_covid19_deaths_global.csv"
    recovered_path = "time_series_covid19_recovered_global.csv"

    # Read datasets
    print("Reading JHU CSSE CSV files...")
    df_confirmed = pd.read_csv(confirmed_path)
    df_deaths = pd.read_csv(deaths_path)
    df_recovered = pd.read_csv(recovered_path)

    # Date columns are all columns starting from index 4 (after Province/State, Country/Region, Lat, Long)
    date_cols = df_confirmed.columns[4:]

    # Parse column headers to pandas datetime to sort and filter correctly
    date_series = pd.to_datetime(date_cols, format="%m/%d/%y")
    
    # Map back dates
    date_mapping = dict(zip(date_series, date_cols))
    sorted_dates = sorted(date_series)

    # Filter dates based on mode
    if mode == "replication":
        print("Running in Replication Mode (Jan 22, 2020 - Apr 6, 2020)")
        start_date = pd.to_datetime("2020-01-22")
        end_date = pd.to_datetime("2020-04-06")
        filtered_dates = [d for d in sorted_dates if start_date <= d <= end_date]
    else:
        print("Running in Extension Mode (Jan 22, 2020 - Mar 9, 2023)")
        filtered_dates = sorted_dates

    # Convert dates back to original string headers
    filtered_date_cols = [date_mapping[d] for d in filtered_dates]

    # Aggregate cumulative counts globally
    global_confirmed_cum = df_confirmed[filtered_date_cols].sum(axis=0)
    global_deaths_cum = df_deaths[filtered_date_cols].sum(axis=0)
    global_recovered_cum = df_recovered[filtered_date_cols].sum(axis=0)

    # Calculate daily new counts (first order difference)
    # The first day's new cases are set to the cumulative count on that day
    global_confirmed_new = global_confirmed_cum.diff().fillna(global_confirmed_cum.iloc[0])
    global_deaths_new = global_deaths_cum.diff().fillna(global_deaths_cum.iloc[0])
    global_recovered_new = global_recovered_cum.diff().fillna(global_recovered_cum.iloc[0])

    # Combine into a single DataFrame
    processed_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in filtered_dates],
        "Confirmed_Cumulative": global_confirmed_cum.values,
        "Deaths_Cumulative": global_deaths_cum.values,
        "Recovered_Cumulative": global_recovered_cum.values,
        "Confirmed_Daily": global_confirmed_new.values.astype(int),
        "Deaths_Daily": global_deaths_new.values.astype(int),
        "Recovered_Daily": global_recovered_new.values.astype(int)
    })

    # Save to CSV
    output_path = "global_covid_daily.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}. Total rows: {len(processed_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess JHU CSSE COVID-19 data.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["replication", "extension"], 
        default="replication",
        help="Replication mode (up to April 6, 2020) or Extension mode (up to March 9, 2023)"
    )
    args = parser.parse_args()
    preprocess_data(mode=args.mode)
