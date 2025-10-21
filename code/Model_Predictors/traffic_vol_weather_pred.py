import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

#added predictor instead of importing as the training code is in different dir
class AdvancedDenseTrafficPredictor:
    def __init__(self, model_path: str = "traffic_adv_model"):
        self.model = None
        self.scaler = None
        self.variance_selector = None
        self.feature_names = []
        self.model_path = model_path

    def preprocess_new(self, data: pd.DataFrame) -> np.ndarray:
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)

        if 'date_time' in data.columns:
            data['date_time'] = pd.to_datetime(data['date_time'], errors='coerce')
            data['hour'] = data['date_time'].dt.hour.fillna(0).astype(int)
            data['day_of_week'] = data['date_time'].dt.dayofweek.fillna(0).astype(int)
            data['month'] = data['date_time'].dt.month.fillna(1).astype(int)
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            data = data.drop(columns=['date_time'], errors='ignore')

        data = data.drop(columns=['weather_description'], errors='ignore')

        for col in data.select_dtypes(include=["object"]).columns:
            data[col] = pd.factorize(data[col])[0]

        if self.variance_selector is not None and self.feature_names:
            all_feats = getattr(self.variance_selector, "feature_names_in_", self.feature_names)
            for f in all_feats:
                if f not in data.columns:
                    data[f] = 0
            data = data[all_feats]
            data_selected = self.variance_selector.transform(data.values)
            X_scaled = self.scaler.transform(data_selected)
        else:
            for f in self.feature_names:
                if f not in data.columns:
                    data[f] = 0
            data = data[self.feature_names]
            X_scaled = self.scaler.transform(data.values)
        return X_scaled

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        X = self.preprocess_new(data)
        preds = self.model.predict(X).flatten()
        return preds


if __name__ == "__main__":
    csv_file = "../Model_Predictors/traffic_data.csv"
    model_file = "../cloud/traffic_adv_model_best.h5"
    pkl_file = "../cloud/traffic_adv_model_preprocessing.pkl"


    df = pd.read_csv(csv_file)
    print(f"[info] Loaded dataset shape: {df.shape}")

    sample = {}
    for col in df.columns:
        if col == "traffic_volume":
            continue
        elif col == "date_time":
            sample[col] = [datetime(2023, 8, 25, 17, 45)]
        elif df[col].dtype == "object":
            sample[col] = [np.random.choice(df[col].dropna().unique())]
        elif np.issubdtype(df[col].dtype, np.number):
            mean, std = df[col].mean(), df[col].std()
            sample[col] = [float(np.random.normal(mean, std))]
        else:
            sample[col] = [df[col].dropna().sample(1).values[0]]

    sample_df = pd.DataFrame(sample)
    print("[info] Input sample for prediction:")
    print(sample_df)

    predictor = AdvancedDenseTrafficPredictor(model_path="../../cloud/traffic_adv_model")


    predictor.model = load_model(model_file)
    print(f"[load] Model loaded from {model_file}")


    with open(pkl_file, "rb") as f:
        prep_data = pickle.load(f)
        predictor.scaler = prep_data["scaler"]
        predictor.variance_selector = prep_data["variance_selector"]
        prep_meta = prep_data.get("prep", {})
        predictor.feature_names = prep_meta.get("feature_names", [])
    print(f"[load] Preprocessing loaded from {pkl_file}")


    prediction = predictor.predict(sample_df)[0]
    print(f"\nPredicted Traffic Volume: {prediction:.2f}")


    sample_df["predicted_traffic_volume"] = prediction
    sample_df.to_csv("single_prediction.csv", index=False)
    print("[io] Saved single sample prediction to single_prediction.csv")
