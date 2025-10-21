"""
Generates realistic synthetic test data that statistically resembles
the original training data (traffic_data.csv).

Output:
    traffic_test_realistic.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_realistic_test_data(train_file="traffic_data.csv", num_days=7):
    print(f"Loading training data from: {train_file}")
    df_train = pd.read_csv(train_file)
    if 'date_time' in df_train.columns:
        df_train['date_time'] = pd.to_datetime(df_train['date_time'], errors='coerce')


    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    stats = df_train[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]

    print("\nNumeric feature stats extracted from training data:")
    print(stats)


    weather_conditions = (
        df_train['weather_main'].dropna().unique().tolist()
        if 'weather_main' in df_train.columns else
        ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
    )
    holidays = (
        df_train['holiday'].dropna().unique().tolist()
        if 'holiday' in df_train.columns else
        [None, 'Christmas', 'New Year']
    )

    start_date = df_train['date_time'].min() if 'date_time' in df_train.columns else datetime(2024, 3, 1)
    end_date = start_date + timedelta(days=num_days)
    date_rng = pd.date_range(start=start_date, end=end_date, freq="H")

    synthetic_data = []
    for dt in date_rng:
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        month = dt.month

        weather = random.choice(weather_conditions)
        holiday = random.choice(holidays if random.random() < 0.1 else [None])


        temp = np.clip(np.random.normal(stats.loc['temp', 'mean'], stats.loc['temp', 'std']), 
                       stats.loc['temp', 'min'], stats.loc['temp', 'max']) if 'temp' in stats.index else 290
        
        rain_1h = np.clip(np.random.normal(stats.loc['rain_1h', 'mean'], stats.loc['rain_1h', 'std']),
                          0, stats.loc['rain_1h', 'max']) if 'rain_1h' in stats.index else 0
        
        snow_1h = np.clip(np.random.normal(stats.loc['snow_1h', 'mean'], stats.loc['snow_1h', 'std']),
                          0, stats.loc['snow_1h', 'max']) if 'snow_1h' in stats.index else 0
        
        clouds_all = int(np.clip(np.random.normal(stats.loc['clouds_all', 'mean'], stats.loc['clouds_all', 'std']),
                                 0, 100)) if 'clouds_all' in stats.index else random.randint(10, 90)

        base = stats.loc['traffic_volume', 'mean'] if 'traffic_volume' in stats.index else 3000
        traffic = base

        if 7 <= hour <= 9 or 17 <= hour <= 19:
            traffic *= random.uniform(1.3, 1.6)
        elif 10 <= hour <= 16:
            traffic *= random.uniform(1.0, 1.2)
        else:
            traffic *= random.uniform(0.5, 0.9)

        if is_weekend:
            traffic *= random.uniform(0.6, 0.85)

        if weather == 'Rain':
            traffic *= random.uniform(0.8, 0.95)
        elif weather == 'Snow':
            traffic *= random.uniform(0.7, 0.9)

        traffic += np.random.normal(0, stats.loc['traffic_volume', 'std'] * 0.3 if 'traffic_volume' in stats.index else 200)

        synthetic_data.append({
            "date_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "holiday": holiday,
            "temp": round(temp, 2),
            "rain_1h": round(rain_1h, 2),
            "snow_1h": round(snow_1h, 2),
            "clouds_all": clouds_all,
            "weather_main": weather,
            "weather_description": f"{weather.lower()} conditions",
            "traffic_volume": int(max(0, traffic))
        })

    df_synth = pd.DataFrame(synthetic_data)
    print(f"\nGenerated {len(df_synth)} samples ({num_days} days of hourly data).")
    df_synth.to_csv("traffic_test_realistic.csv", index=False)
    print("Saved to traffic_test_realistic.csv\n")
    print(df_synth.head(10))
    return df_synth


if __name__ == "__main__":
    generate_realistic_test_data("traffic_data.csv", num_days=7)
