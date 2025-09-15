import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


def create_sequences(data, seq_length=24, target_col="Vehicles"):
    sequences = []
    targets = []
    
    feature_cols = ["Vehicles", "Hour", "DayOfWeek", "Month", "Junction"]

    for junction in data["Junction"].unique():
        junction_data = data[data["Junction"] == junction].copy()
        junction_data = junction_data.sort_values("DateTime").reset_index(drop=True)

        features = junction_data[feature_cols].values
        
        for i in range(seq_length, len(junction_data)):
            sequences.append(features[i-seq_length:i])
            targets.append(junction_data[target_col].iloc[i])
    
    return np.array(sequences), np.array(targets)


model = load_model("traffic_prediction_cnn_model.h5")
print("Model loaded successfully")


train_df = pd.read_csv("traffic.csv")
train_df["DateTime"] = pd.to_datetime(train_df["DateTime"])


train_df["Hour"] = train_df["DateTime"].dt.hour
train_df["DayOfWeek"] = train_df["DateTime"].dt.dayofweek
train_df["Month"] = train_df["DateTime"].dt.month


X_train_full, y_train_full = create_sequences(train_df, seq_length=24)


scaler_X = MinMaxScaler().fit(X_train_full.reshape(-1, X_train_full.shape[2]))
scaler_y = MinMaxScaler().fit(y_train_full.reshape(-1, 1))

print(" Scalers re-fitted from training data")


df_test = pd.read_csv("traffic_test.csv")
df_test["DateTime"] = pd.to_datetime(df_test["DateTime"])

df_test["Hour"] = df_test["DateTime"].dt.hour
df_test["DayOfWeek"] = df_test["DateTime"].dt.dayofweek
df_test["Month"] = df_test["DateTime"].dt.month


X_test, y_test = create_sequences(df_test, seq_length=24)


X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"Test data shape: {X_test_scaled.shape}")


y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()


print("\nSample Predictions:")
for i in range(10):
    print(f"Sample {i+1}: Actual = {y_actual[i]:.1f}, Predicted = {y_pred[i]:.1f}")
