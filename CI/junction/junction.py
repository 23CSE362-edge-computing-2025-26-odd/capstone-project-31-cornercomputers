import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('traffic.csv')
print(f"dataset shape: {df.shape}")
print("\nfirst few rows:")
print(df.head())
print("\ndataset info:")
print(df.info())
print("\nbasic stats:")
print(df.describe())

print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

df['DateTime'] = pd.to_datetime(df['DateTime'])

df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Year'] = df['DateTime'].dt.year

print(f"\nmissing values:\n{df.isnull().sum()}")

print(f"\nUnique Junctions: {df['Junction'].unique()}")
print(f"Number of unique junctions: {df['Junction'].nunique()}")

print("\nVehicle count statistics by Junction:")
print(df.groupby('Junction')['Vehicles'].describe())

df = df.sort_values(['Junction', 'DateTime']).reset_index(drop=True)

plt.figure(figsize=(15, 10))

# Plot 1: Traffic by hour of day
plt.subplot(2, 3, 1)
hourly_traffic = df.groupby('Hour')['Vehicles'].mean()
plt.plot(hourly_traffic.index, hourly_traffic.values, marker='o')
plt.title('Average Traffic by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Average Vehicles')
plt.grid(True)

# Plot 2: Traffic by day of week
plt.subplot(2, 3, 2)
weekly_traffic = df.groupby('DayOfWeek')['Vehicles'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.bar(range(7), weekly_traffic.values)
plt.title('Average Traffic by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Vehicles')
plt.xticks(range(7), days)
plt.grid(True, alpha=0.3)

# Plot 3: Traffic by junction
plt.subplot(2, 3, 3)
junction_traffic = df.groupby('Junction')['Vehicles'].mean()
plt.bar(junction_traffic.index, junction_traffic.values, color='orange')
plt.title('Average Traffic by Junction')
plt.xlabel('Junction')
plt.ylabel('Average Vehicles')
plt.grid(True, alpha=0.3)

# Plot 4: Time series for each junction
plt.subplot(2, 3, 4)
for junction in df['Junction'].unique():
    junction_data = df[df['Junction'] == junction].copy()
    junction_data = junction_data.sort_values('DateTime')
    plt.plot(junction_data['DateTime'], junction_data['Vehicles'], 
             label=f'Junction {junction}', alpha=0.7)
plt.title('Traffic Time Series by Junction')
plt.xlabel('Date')
plt.ylabel('Vehicles')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 5: Distribution of vehicle counts
plt.subplot(2, 3, 5)
plt.hist(df['Vehicles'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribution of Vehicle Counts')
plt.xlabel('Vehicles')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Plot 6: Correlation heatmap
plt.subplot(2, 3, 6)
correlation_data = df[['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'Junction']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("PREPARING DATA FOR CNN")
print("="*50)

def create_sequences(data, seq_length, target_col='Vehicles'):
    sequences = []
    targets = []
    
    for junction in data['Junction'].unique():
        junction_data = data[data['Junction'] == junction].copy()
        junction_data = junction_data.sort_values('DateTime').reset_index(drop=True)
        
        feature_cols = ['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'Junction']
        features = junction_data[feature_cols].values
        
        for i in range(seq_length, len(junction_data)):
            sequences.append(features[i-seq_length:i])
            targets.append(junction_data[target_col].iloc[i])
    
    return np.array(sequences), np.array(targets)


SEQUENCE_LENGTH = 24  

print(f"Creating sequences with length: {SEQUENCE_LENGTH}")
X, y = create_sequences(df, SEQUENCE_LENGTH)

print(f"Sequences shape: {X.shape}")
print(f"Targets shape: {y.shape}")


scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()


n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)
X_scaled_reshaped = scaler_X.fit_transform(X_reshaped)
X_scaled = X_scaled_reshaped.reshape(n_samples, n_timesteps, n_features)

y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=False
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")


print("\n" + "="*50)
print("BUILDING CNN MODEL")
print("="*50)

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', 
               input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  
    ])
    
    return model


input_shape = (X_train.shape[1], X_train.shape[2])
model = create_cnn_model(input_shape)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(model.summary())


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=0.0001,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

print("\n" + "="*50)
print("TRAINING THE MODEL")
print("="*50)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("EVALUATING THE MODEL")
print("="*50)

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()


mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)

print(f"Test Metrics:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(15, 5))

# Plot 1: pred vs actual
plt.subplot(1, 3, 1)
plt.scatter(y_test_actual, y_pred, alpha=0.6)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Vehicles')
plt.ylabel('Predicted Vehicles')
plt.title(f'Predictions vs Actual\nR² = {r2:.4f}')
plt.grid(True, alpha=0.3)

# Plot 2: time series
plt.subplot(1, 3, 2)
n_samples_to_plot = min(200, len(y_test_actual))
plt.plot(range(n_samples_to_plot), y_test_actual[:n_samples_to_plot], 
         label='Actual', linewidth=2)
plt.plot(range(n_samples_to_plot), y_pred[:n_samples_to_plot], 
         label='Predicted', linewidth=2, alpha=0.8)
plt.xlabel('Time Steps')
plt.ylabel('Vehicles')
plt.title(f'Time Series Comparison\n(First {n_samples_to_plot} samples)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: residual
plt.subplot(1, 3, 3)
residuals = y_test_actual - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Vehicles')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


def predict_traffic(model, last_sequence, scaler_X, scaler_y, steps_ahead=1):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps_ahead):
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        pred_scaled = model.predict(input_seq, verbose=0)
        pred_actual = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        predictions.append(pred_actual)
        
        new_row = current_sequence[-1].copy()
        new_row[0] = pred_scaled[0][0] 
        
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return predictions

last_seq = X_test[-1]
future_predictions = predict_traffic(model, last_seq, scaler_X, scaler_y, steps_ahead=24)

print(f"\nNext 24 hours predictions:")
for i, pred in enumerate(future_predictions):
    print(f"Hour {i+1}: {pred:.1f} vehicles")


model.save('traffic_prediction_cnn_model.h5')
print(f"\nModel saved as 'traffic_prediction_cnn_model.h5'")

print("\n" + "="*50)
print("MODEL ANALYSIS")
print("="*50)

print("Model Architecture Summary:")
print(f"- Input Shape: {input_shape}")
print(f"- Total Parameters: {model.count_params():,}")
print(f"- Conv1D Layers: 3")
print(f"- Dense Layers: 3")
print(f"- Dropout Layers: 5")
print(f"- BatchNormalization Layers: 3")

print("\nFeature Information:")
feature_names = ['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'Junction']
for i, name in enumerate(feature_names):
    print(f"Feature {i}: {name}")

print(f"\nModel Performance Summary:")
print(f"- Training completed in {len(history.history['loss'])} epochs")
print(f"- Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"- Final test RMSE: {rmse:.2f} vehicles")
print(f"- Model explains {r2*100:.1f}% of the variance in traffic data")