import os
import pandas as pd
import numpy as np

def preprocess_multimodal_data():
    print("Starting Multimodal Data Preprocessing...")
    
    # 1. Paths to raw datasets in base workspace directory
    confirmed_path = os.path.join("..", "time_series_covid19_confirmed_global.csv")
    deaths_path = os.path.join("..", "time_series_covid19_deaths_global.csv")
    recovered_path = os.path.join("..", "time_series_covid19_recovered_global.csv")
    mobility_path = os.path.join("..", "Global_Mobility_Report.csv")
    stringency_path = os.path.join("..", "OxCGRT_timeseries_StringencyIndex_v1.csv")

    # =========================================================================
    # Step A: Load and Process JHU Case Counts (Global Aggregation)
    # =========================================================================
    print("Loading and aggregating JHU case counts...")
    df_conf = pd.read_csv(confirmed_path)
    df_death = pd.read_csv(deaths_path)
    df_rec = pd.read_csv(recovered_path)

    date_cols = df_conf.columns[4:]
    date_series = pd.to_datetime(date_cols, format="%m/%d/%y")
    date_mapping = dict(zip(date_series, date_cols))
    sorted_dates = sorted(date_series)
    sorted_date_cols = [date_mapping[d] for d in sorted_dates]

    # Global daily cumulative counts
    global_conf = df_conf[sorted_date_cols].sum(axis=0).values
    global_death = df_death[sorted_date_cols].sum(axis=0).values
    global_rec = df_rec[sorted_date_cols].sum(axis=0).values

    # Daily new counts
    daily_conf = np.diff(global_conf, prepend=global_conf[0])
    daily_death = np.diff(global_death, prepend=global_death[0])
    daily_rec = np.diff(global_rec, prepend=global_rec[0])

    jhu_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in sorted_dates],
        "Confirmed_Cumulative": global_conf,
        "Deaths_Cumulative": global_death,
        "Recovered_Cumulative": global_rec,
        "Confirmed_Daily": daily_conf,
        "Deaths_Daily": daily_death,
        "Recovered_Daily": daily_rec
    })
    
    # Fill any negative daily entries caused by JHU revisions
    jhu_df["Confirmed_Daily"] = jhu_df["Confirmed_Daily"].clip(lower=0)
    jhu_df["Deaths_Daily"] = jhu_df["Deaths_Daily"].clip(lower=0)
    jhu_df["Recovered_Daily"] = jhu_df["Recovered_Daily"].clip(lower=0)

    # =========================================================================
    # Step B: Stream & Process Google Mobility Index (Optimized)
    # =========================================================================
    print("Loading and optimizing Google Mobility Report...")
    # Read only required columns to save RAM
    use_cols = [
        "sub_region_1", "date", 
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline"
    ]
    
    # Filter out sub-regions (keeping only national level records) during loading
    chunks = []
    for chunk in pd.read_csv(mobility_path, usecols=use_cols, chunksize=100000, low_memory=False):
        # Keep national rows (sub_region_1 is null)
        national_chunk = chunk[chunk["sub_region_1"].isna()]
        chunks.append(national_chunk)
    
    df_mob = pd.concat(chunks, axis=0)
    df_mob["date"] = pd.to_datetime(df_mob["date"])
    
    # Average across all countries per date to get a global mobility proxy
    mobility_features = [col for col in use_cols if col not in ["sub_region_1", "date"]]
    global_mob = df_mob.groupby("date")[mobility_features].mean().reset_index()
    global_mob["Date"] = global_mob["date"].dt.strftime("%Y-%m-%d")
    global_mob = global_mob.drop(columns=["date"])

    # =========================================================================
    # Step C: Load and Parse Oxford Stringency Index
    # =========================================================================
    print("Loading and parsing Oxford Stringency Index...")
    df_str = pd.read_csv(stringency_path)
    
    # Known metadata (non-date) columns based on actual file inspection
    meta_cols = ["CountryCode", "CountryName", "RegionCode", "RegionName",
                 "CityCode", "CityName", "Jurisdiction"]
    existing_meta = [c for c in meta_cols if c in df_str.columns]
    
    # Keep only national-level rows to avoid double-counting sub-regions
    if "Jurisdiction" in df_str.columns:
        df_str = df_str[df_str["Jurisdiction"] == "NAT_TOTAL"]
    
    # Date columns are all remaining columns (format is like '01Jan2020')
    date_cols_str = [c for c in df_str.columns if c not in existing_meta]
    
    # Melt wide format to long format
    df_str_melted = pd.melt(df_str, id_vars=existing_meta, value_vars=date_cols_str, 
                            var_name="Date_Raw", value_name="StringencyIndex")
    
    # Parse Oxford date format exactly: e.g. '01Jan2020'
    df_str_melted["Date"] = pd.to_datetime(
        df_str_melted["Date_Raw"], format="%d%b%Y"
    ).dt.strftime("%Y-%m-%d")
    
    # Average stringency indices globally per date
    global_str = df_str_melted.groupby("Date")["StringencyIndex"].mean().reset_index()

    # =========================================================================
    # Step D: Align and Merge All Datasets
    # =========================================================================
    print("Merging and aligning multi-modal data...")
    merged_df = pd.merge(jhu_df, global_mob, on="Date", how="left")
    merged_df = pd.merge(merged_df, global_str, on="Date", how="left")

    # Forward-fill any missing mobility or policy dates (e.g. early Jan 2020)
    # and fill remaining NaNs with 0 (baseline)
    fill_cols = mobility_features + ["StringencyIndex"]
    merged_df[fill_cols] = merged_df[fill_cols].ffill().fillna(0)

    # Save to output file
    output_filename = "multimodal_aligned_data.csv"
    merged_df.to_csv(output_filename, index=False)
    print(f"Aligning completed! Dataset saved to {output_filename}. Total records: {len(merged_df)}")

if __name__ == "__main__":
    preprocess_multimodal_data()
