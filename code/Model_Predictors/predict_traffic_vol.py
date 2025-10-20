import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# Configuration
CSV_FILE = "traffic_test.csv"
MODEL_FILE = "../cloud/traffic_cnn_model_best.h5"
SEQ_LENGTH = 10

def load_traffic_data(csv_file):
    """Load traffic data from CSV file"""
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Creating synthetic test data instead...")
        return create_synthetic_data()
    
    df = pd.read_csv(csv_file)
    print(f"Data loaded: {df.shape[0]} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    return df

def create_synthetic_data():
    """Create synthetic test data if CSV doesn't exist"""
    num_days = 5
    junctions = [1]
    start_date = "2025-03-01"
    
    date_rng = pd.date_range(start=start_date, periods=num_days*24, freq="h")
    
    data = []
    for j in junctions:
        base = np.random.randint(50, 200)
        for dt in date_rng:
            hour = dt.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                vehicles = base + np.random.randint(40, 80)
            elif 0 <= hour <= 5:
                vehicles = base + np.random.randint(-30, -10)
            else:
                vehicles = base + np.random.randint(-10, 20)
            vehicles = max(vehicles, 0)
            data.append([dt, j, vehicles])
    
    df = pd.DataFrame(data, columns=["DateTime", "Junction", "Vehicles"])
    print(f"Synthetic data created: {df.shape[0]} rows")
    return df

def preprocess_data(df, seq_length=10):
    """Preprocess traffic data"""
    # Convert DateTime to datetime format if it's a string
    if df["DateTime"].dtype == 'object':
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    
    # Sort by Junction and DateTime
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    
    print(f"\nData preprocessed")
    print(f"Unique Junctions: {df['Junction'].unique()}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    
    return df

def create_sequences(data, seq_length=10, target_col="Vehicles"):
    """Create sequences for CNN model (simplified - uses only Vehicles column)"""
    sequences, targets = [], []
    
    for junction in data["Junction"].unique():
        junc_data = data[data["Junction"] == junction].sort_values("DateTime").reset_index(drop=True)
        
        # Check if we have enough data
        if len(junc_data) < seq_length:
            print(f"Warning: Junction {junction} has only {len(junc_data)} records (need at least {seq_length})")
            continue
            
        # Extract Vehicles column
        features = junc_data[[target_col]].values
        
        for i in range(seq_length, len(junc_data)):
            sequences.append(features[i-seq_length:i])
            targets.append(junc_data[target_col].iloc[i])
    
    return np.array(sequences), np.array(targets)

def scale_data(X, y):
    """Scale data using MinMaxScaler"""
    # For this model, X is (samples, seq_length, 1)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Fit and transform X
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(n_samples, n_timesteps, n_features)
    
    # Fit and transform y
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    print(f"\nData scaled")
    print(f"X_scaled shape: {X_scaled.shape}")
    print(f"y_scaled shape: {y_scaled.shape}")
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def make_predictions(model, X_scaled, scaler_y):
    """Make predictions using the loaded model"""
    print(f"\nModel input shape expected: {model.input_shape}")
    print(f"X_scaled shape: {X_scaled.shape}")
    
    # Make predictions
    try:
        y_pred_scaled = model.predict(X_scaled, verbose=0)
    except Exception as e:
        print(f"Warning: Error during prediction: {e}")
        print("Attempting to predict smaller batch...")
        y_pred_scaled = model.predict(X_scaled[:100], verbose=0)
        if len(X_scaled) > 100:
            print(f"Note: Only predicting first 100 samples due to model issue")
            X_scaled = X_scaled[:100]
    
    # Handle output shape
    if len(y_pred_scaled.shape) > 1 and y_pred_scaled.shape[1] == 1:
        y_pred_scaled = y_pred_scaled.flatten()
    
    # Inverse scale predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    print(f"Predictions made: {y_pred.shape}")
    
    return y_pred

def main():
    """Main execution function"""
    print("=" * 60)
    print("Traffic Volume Prediction using CNN Model")
    print("(TrafficVolPred Folder)")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(CSV_FILE):
        print(f"Note: {CSV_FILE} not found. Will generate synthetic data.")
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found!")
        available_models = [f for f in os.listdir('.') if f.endswith('.h5')]
        if available_models:
            print(f"Available .h5 files: {available_models}")
            print(f"Please update MODEL_FILE to one of these: {available_models[0]}")
        return
    
    # Load data
    df = load_traffic_data(CSV_FILE)
    
    # Preprocess data
    df = preprocess_data(df, seq_length=SEQ_LENGTH)
    
    # Create sequences
    X, y = create_sequences(df, seq_length=SEQ_LENGTH)
    print(f"\nSequences created: {X.shape[0]} sequences")
    
    if X.shape[0] == 0:
        print("Error: No sequences created! Check your data and sequence length.")
        return
    
    # Scale data
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data(X, y)
    
    # Load model
    print(f"\nLoading model: {MODEL_FILE}")
    try:
        model = load_model(MODEL_FILE)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model with default settings: {e}")
        print("Attempting to load with custom_objects...")
        try:
            # Try with various custom object configurations
            custom_objs = {
                'mse': 'mean_squared_error',
                'mae': 'mean_absolute_error',
            }
            model = load_model(MODEL_FILE, custom_objects=custom_objs)
            print("Model loaded with custom objects!")
        except Exception as e2:
            print(f"Still having issues. Trying compile=False...")
            try:
                model = load_model(MODEL_FILE, compile=False)
                print("Model loaded without recompiling!")
            except Exception as e3:
                print(f"Error: Could not load model: {e3}")
                return
    
    # Make predictions
    y_pred = make_predictions(model, X_scaled, scaler_y)
    
    # Display results
    print("\n" + "=" * 60)
    print("Prediction Results (First 20 samples):")
    print("=" * 60)
    print(f"{'Sample':<10} {'Predicted':<15} {'Scaled Actual':<15}")
    print("-" * 60)
    
    display_count = min(20, len(y_pred))
    for i in range(display_count):
        print(f"{i+1:<10} {y_pred[i]:<15.2f} {y_scaled[i]:<15.4f}")
    
    print("=" * 60)
    print(f"\nPrediction Statistics:")
    print(f"Mean Prediction: {y_pred.mean():.2f}")
    print(f"Min Prediction: {y_pred.min():.2f}")
    print(f"Max Prediction: {y_pred.max():.2f}")
    print(f"Std Dev: {y_pred.std():.2f}")
    
    # Calculate accuracy metrics if we have actual values
    print(f"\nScaled Data Statistics:")
    print(f"Mean Actual: {y_scaled.mean():.4f}")
    print(f"Min Actual: {y_scaled.min():.4f}")
    print(f"Max Actual: {y_scaled.max():.4f}")
    
    # Calculate MAE and RMSE
    mae = np.mean(np.abs(y_pred - scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()))
    rmse = np.sqrt(np.mean((y_pred - scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten())**2))
    print(f"\nModel Performance:")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    
    # Save results
    results_file = "traffic_predictions.csv"
    results_df = pd.DataFrame({
        'Prediction': y_pred,
        'Scaled_Actual': y_scaled[:len(y_pred)],
        'Actual': scaler_y.inverse_transform(y_scaled[:len(y_pred)].reshape(-1, 1)).flatten()
    })
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
