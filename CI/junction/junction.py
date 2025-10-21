"""
Changes:
 - 1 hot encoding for junction column
 - L2 regularization over drouput in 1convD
 - casual padding
 - new scaler is used single across both x and y
 - reLu activation over linear
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('traffic.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values(['Junction', 'DateTime']).reset_index(drop=True)

df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Year'] = df['DateTime'].dt.year

junction_dummies = pd.get_dummies(df['Junction'].astype(str), prefix='J')
df = pd.concat([df, junction_dummies], axis=1)

VEH_COL = 'Vehicles'
FEATURE_COLS = ['Vehicles', 'Hour', 'DayOfWeek', 'Month'] + list(junction_dummies.columns)
SEQ_LEN = 24

scaler = MinMaxScaler()
scaler.fit(df[FEATURE_COLS])
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
hourly = df.groupby('Hour')['Vehicles'].mean()
plt.plot(hourly.index, hourly.values, marker='o')
plt.title('Average Traffic by Hour')
plt.xlabel('Hour')
plt.ylabel('Vehicles')

plt.subplot(2, 3, 2)
weekly = df.groupby('DayOfWeek')['Vehicles'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.bar(range(7), weekly.values)
plt.title('Avg Traffic by Day of Week')
plt.xticks(range(7), days)

plt.subplot(2, 3, 3)
junction_avg = df.groupby('Junction')['Vehicles'].mean()
plt.bar(junction_avg.index, junction_avg.values, color='orange')
plt.title('Avg Traffic by Junction')
plt.xlabel('Junction')

plt.subplot(2, 3, 4)
for j in df['Junction'].unique():
    subset = df[df['Junction'] == j]
    plt.plot(subset['DateTime'], subset['Vehicles'], label=f'Junction {j}', alpha=0.7)
plt.legend()
plt.title('Traffic Time Series')
plt.xlabel('Date')
plt.ylabel('Vehicles')

plt.subplot(2, 3, 5)
plt.hist(df['Vehicles'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Vehicle Counts')
plt.xlabel('Vehicles')

plt.subplot(2, 3, 6)
corr = df[['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'Junction']].corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation')
plt.tight_layout()
plt.show()

def create_sequences(data, seq_len=24):
    X, y = [], []
    for j in data['Junction'].unique():
        jd = data[data['Junction'] == j].sort_values('DateTime').reset_index(drop=True)
        if len(jd) <= seq_len:
            continue
        vals = jd[FEATURE_COLS].values
        for i in range(seq_len, len(jd)):
            X.append(vals[i-seq_len:i])
            y.append(vals[i, 0])
    return np.array(X), np.array(y).reshape(-1, 1)

X, y = create_sequences(df_scaled, SEQ_LEN)
print(f"\nData prepared: X shape {X.shape}, y shape {y.shape}")

n_total = len(X)
n_test = int(0.2 * n_total)
n_val = int(0.2 * (n_total - n_test))

X_train = X[:-(n_val + n_test)]
y_train = y[:-(n_val + n_test)]
X_val = X[-(n_val + n_test):-n_test]
y_val = y[-(n_val + n_test):-n_test]
X_test = X[-n_test:]
y_test = y[-n_test:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='causal', kernel_regularizer=l2(1e-4), input_shape=input_shape),
        Conv1D(128, 3, activation='relu', padding='causal', kernel_regularizer=l2(1e-4)),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 3, activation='relu', padding='causal', kernel_regularizer=l2(1e-4)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='relu')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
                  loss='mse', metrics=['mae'])
    return model

model = build_cnn((X_train.shape[1], X_train.shape[2]))
print("\nModel Summary:")
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5)
callbacks = [early_stop, reduce_lr]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Model Loss')
plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend(); plt.title('Model MAE')
plt.tight_layout(); plt.show()

y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(np.hstack([y_pred_scaled, np.zeros((len(y_pred_scaled), len(FEATURE_COLS)-1))]))[:,0]
y_pred = np.clip(y_pred, 0, None)
y_test_raw = scaler.inverse_transform(np.hstack([y_test, np.zeros((len(y_test), len(FEATURE_COLS)-1))]))[:,0]

mae = mean_absolute_error(y_test_raw, y_pred)
mse = mean_squared_error(y_test_raw, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_raw, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.4f}")
print("="*50)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(y_test_raw, y_pred, alpha=0.6)
plt.plot([y_test_raw.min(), y_test_raw.max()],
         [y_test_raw.min(), y_test_raw.max()], 'r--')
plt.title(f"Predicted vs Actual (R²={r2:.3f})")
plt.xlabel("Actual Vehicles")
plt.ylabel("Predicted Vehicles")
plt.grid(True, alpha=0.3)

plt.subplot(1,3,2)
n_show = min(200, len(y_test_raw))
plt.plot(range(n_show), y_test_raw[:n_show], label="Actual")
plt.plot(range(n_show), y_pred[:n_show], label="Predicted", alpha=0.8)
plt.title("Time Series Comparison (First 200 samples)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1,3,3)
residuals = y_test_raw - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

corr_coef = np.corrcoef(y_test_raw, y_pred)[0,1]
print(f"\nCorrelation between actual and predicted: {corr_coef:.4f}")

def predict_future(model, last_seq_scaled, steps=24):
    seq = last_seq_scaled.copy()
    preds = []
    for _ in range(steps):
        inp = seq.reshape(1, seq.shape[0], seq.shape[1])
        pred_scaled = model.predict(inp, verbose=0)
        preds.append(pred_scaled[0,0])
        new_row = seq[-1].copy()
        new_row[0] = pred_scaled[0,0]
        seq = np.vstack([seq[1:], new_row])
    preds_raw = scaler.inverse_transform(
        np.hstack([np.array(preds).reshape(-1,1), np.zeros((len(preds), len(FEATURE_COLS)-1))])
    )[:,0]
    preds_raw = np.clip(preds_raw, 0, None)
    return preds_raw

last_seq = X_test[-1]
future_preds = predict_future(model, last_seq, steps=24)

print("\nNext 24-hour traffic volume predictions:")
for i, val in enumerate(future_preds, 1):
    print(f"Hour {i}: {val:.1f} vehicles")

print("\n" + "="*50)
print("MODEL ANALYSIS SUMMARY")
print("="*50)
print(f"- Input Shape: {X_train.shape[1:]}") 
print(f"- Total Parameters: {model.count_params():,}")
print(f"- Conv Layers: 3")
print(f"- Dense Layers: 3")
print(f"- Regularization: L2 (1e-4)")
print(f"- Output Activation: ReLU (Non-negative)")
print(f"- Final Test RMSE: {rmse:.2f}")
print(f"- Model explains {r2*100:.1f}% variance in data")

model.save('traffic_cnn_best_model.h5')
joblib.dump(scaler, 'traffic_scaler.save')
print("\nModel and scaler saved successfully.")
