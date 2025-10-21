import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime, timedelta
import argparse

# Configuration
CSV_FILE = "traffic.csv"
MODEL_FILE = "../cloud/traffic_cnn_best_model.h5"
SCALER_FILE = "../cloud/traffic_scaler.save"
SEQ_LEN = 24

def load_reference_data():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"[error] {CSV_FILE} not found.")
    df = pd.read_csv(CSV_FILE)
    stats = df.groupby("Junction")["Vehicles"].agg(["mean", "std"]).to_dict("index")
    all_junctions = sorted(df["Junction"].unique().tolist())
    return stats, all_junctions

def generate_synthetic_junction_data(junction_id: int, days: int = 7):
    stats, all_junctions = load_reference_data()
    if junction_id not in stats:
        raise ValueError(f"Junction {junction_id} not found in dataset. Available: {all_junctions}")
    
    mean_val = stats[junction_id]["mean"]
    std_val = stats[junction_id]["std"]
    print(f"[data] Generating synthetic data for Junction {junction_id} (μ={mean_val:.2f}, σ={std_val:.2f})")


    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    date_rng = pd.date_range(start=start_date, end=end_date, freq="h")

    vehicles = []
    for dt in date_rng:
        hour = dt.hour
        noise = np.random.normal(0, std_val * 0.3)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            val = mean_val + std_val * 0.6 + noise
        elif 0 <= hour <= 5:
            val = mean_val - std_val * 0.4 + noise
        else:
            val = mean_val + noise
        vehicles.append(max(0, val))
    
    df = pd.DataFrame({
        "DateTime": date_rng,
        "Junction": [junction_id] * len(date_rng),
        "Vehicles": vehicles
    })
    df.to_csv(f"junction_{junction_id}_synthetic.csv", index=False)
    print(f"[data] Synthetic data saved → junction_{junction_id}_synthetic.csv ({len(df)} rows)")
    return df, all_junctions

def preprocess_data(df, all_junctions):
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Month"] = df["DateTime"].dt.month

    junction_dummies = pd.get_dummies(df["Junction"].astype(str), prefix="J")


    for j in all_junctions:
        col_name = f"J_{j}"
        if col_name not in junction_dummies.columns:
            junction_dummies[col_name] = 0
    junction_dummies = junction_dummies[[f"J_{j}" for j in all_junctions]]

    df = pd.concat([df, junction_dummies], axis=1)
    feature_cols = ["Vehicles", "Hour", "DayOfWeek", "Month"] + list(junction_dummies.columns)
    return df, feature_cols

def predict_next_24h(model, scaler, df, feature_cols):
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    last_seq = df_scaled[feature_cols].values[-SEQ_LEN:]
    seq = last_seq.copy()
    preds_scaled = []

    for _ in range(24):
        inp = seq.reshape(1, SEQ_LEN, len(feature_cols))
        pred_scaled = model.predict(inp, verbose=0)[0, 0]
        preds_scaled.append(pred_scaled)
        new_row = seq[-1].copy()
        new_row[0] = pred_scaled
        seq = np.vstack([seq[1:], new_row])
    
    preds_raw = scaler.inverse_transform(
        np.hstack([np.array(preds_scaled).reshape(-1, 1), np.zeros((24, len(feature_cols) - 1))])
    )[:, 0]
    preds_raw = np.clip(preds_raw, 0, None)
    return preds_raw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--junction", type=int, default=1, help="Junction ID (1–4)")
    args = parser.parse_args()
    junction_id = args.junction

    print("=" * 65)
    print(f"Traffic Volume Prediction (Next 24 Hours) — Junction {junction_id}")
    print("=" * 65)

    if not os.path.exists(MODEL_FILE):
        print(f"[error] Model file {MODEL_FILE} not found.")
        return
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("[load] Model and scaler loaded successfully.")

    df, all_junctions = generate_synthetic_junction_data(junction_id)
    df, feature_cols = preprocess_data(df, all_junctions)

    preds = predict_next_24h(model, scaler, df, feature_cols)

    future_times = pd.date_range(df["DateTime"].iloc[-1] + timedelta(hours=1), periods=24, freq="h")
    results = pd.DataFrame({"DateTime": future_times, "Predicted_Vehicles": preds})
    results.to_csv(f"junction_{junction_id}_next24h.csv", index=False)

    print(f"[pred] Next 24-hour forecast generated for Junction {junction_id}")
    print(f"[save] Saved → junction_{junction_id}_next24h.csv")
    print("\nSample:")
    print(results.head())

if __name__ == "__main__":
    main()
