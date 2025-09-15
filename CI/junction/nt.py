import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import warnings
warnings.filterwarnings('ignore')

class TrafficPredictor:
    def __init__(self, model_path='traffic_prediction_cnn_model.h5'):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 24
        self.feature_names = ['Vehicles', 'Hour', 'DayOfWeek', 'Month', 'Junction']
        self.load_model(model_path)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    
    def load_model(self, model_path):
        try:
            self.model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Model input shape: {self.model.input_shape}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model file exists and is valid.")
    
    def preprocess_data(self, df):
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data['DateTime']):
            data['DateTime'] = pd.to_datetime(data['DateTime'])
        data['Hour'] = data['DateTime'].dt.hour
        data['DayOfWeek'] = data['DateTime'].dt.dayofweek
        data['Month'] = data['DateTime'].dt.month
        data['Day'] = data['DateTime'].dt.day
        data['Year'] = data['DateTime'].dt.year
        data = data.sort_values(['Junction', 'DateTime']).reset_index(drop=True)
        print(f"Preprocessed data shape: {data.shape}")
        print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")
        print(f"Junctions: {sorted(data['Junction'].unique())}")
        return data
    
    def create_sequences(self, data):
        sequences = []
        metadata = []
        for junction in data['Junction'].unique():
            junction_data = data[data['Junction'] == junction].copy()
            junction_data = junction_data.sort_values('DateTime').reset_index(drop=True)
            if len(junction_data) < self.sequence_length:
                print(f"Warning: Junction {junction} has only {len(junction_data)} records, need at least {self.sequence_length} for prediction")
                continue
            features = junction_data[self.feature_names].values
            for i in range(self.sequence_length, len(junction_data)):
                sequences.append(features[i-self.sequence_length:i])
                metadata.append({
                    'junction': junction,
                    'datetime': junction_data['DateTime'].iloc[i],
                    'actual_vehicles': junction_data['Vehicles'].iloc[i] if 'Vehicles' in junction_data.columns else None,
                    'sequence_start': junction_data['DateTime'].iloc[i-self.sequence_length],
                    'sequence_end': junction_data['DateTime'].iloc[i-1]
                })
        if len(sequences) == 0:
            raise ValueError("No valid sequences could be created from the data")
        return np.array(sequences), metadata
    
    def fit_scalers(self, data):
        sequences, _ = self.create_sequences(data)
        n_samples, n_timesteps, n_features = sequences.shape
        X_reshaped = sequences.reshape(-1, n_features)
        self.scaler_X.fit(X_reshaped)
        vehicles_data = data['Vehicles'].values.reshape(-1, 1)
        self.scaler_y.fit(vehicles_data)
        print("Scalers fitted on new data")
    
    def predict_sequences(self, data):
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        processed_data = self.preprocess_data(data)
        self.fit_scalers(processed_data)
        sequences, metadata = self.create_sequences(processed_data)
        n_samples, n_timesteps, n_features = sequences.shape
        X_reshaped = sequences.reshape(-1, n_features)
        X_scaled_reshaped = self.scaler_X.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape(n_samples, n_timesteps, n_features)
        print(f"Making predictions on {len(sequences)} sequences...")
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        results = pd.DataFrame(metadata)
        results['predicted_vehicles'] = y_pred
        if results['actual_vehicles'].notna().any():
            results['prediction_error'] = results['actual_vehicles'] - results['predicted_vehicles']
            results['absolute_error'] = abs(results['prediction_error'])
            results['percentage_error'] = (results['prediction_error'] / results['actual_vehicles']) * 100
        return results
    
    def predict_future(self, data, junction_id, hours_ahead=24):
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        processed_data = self.preprocess_data(data)
        self.fit_scalers(processed_data)
        junction_data = processed_data[processed_data['Junction'] == junction_id].copy()
        junction_data = junction_data.sort_values('DateTime').reset_index(drop=True)
        if len(junction_data) < self.sequence_length:
            raise ValueError(f"Not enough data for junction {junction_id}. Need at least {self.sequence_length} records.")
        features = junction_data[self.feature_names].values
        last_sequence = features[-self.sequence_length:]
        last_sequence_reshaped = last_sequence.reshape(-1, len(self.feature_names))
        last_sequence_scaled_reshaped = self.scaler_X.transform(last_sequence_reshaped)
        last_sequence_scaled = last_sequence_scaled_reshaped.reshape(self.sequence_length, len(self.feature_names))
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        last_datetime = junction_data['DateTime'].iloc[-1]
        for i in range(hours_ahead):
            input_seq = current_sequence.reshape(1, self.sequence_length, len(self.feature_names))
            pred_scaled = self.model.predict(input_seq, verbose=0)
            pred_actual = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            future_datetime = last_datetime + pd.Timedelta(hours=i+1)
            predictions.append({
                'datetime': future_datetime,
                'predicted_vehicles': pred_actual,
                'hour': future_datetime.hour,
                'day_of_week': future_datetime.dayofweek,
                'junction': junction_id
            })
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_scaled[0][0]
            new_row[1] = (future_datetime.hour) / 23.0
            new_row[2] = (future_datetime.dayofweek) / 6.0
            new_row[3] = (future_datetime.month - 1) / 11.0
            current_sequence = np.vstack([current_sequence[1:], new_row])
        return predictions
    
    def visualize_predictions(self, results, junction_id=None, max_points=200):
        if junction_id is not None:
            plot_data = results[results['junction'] == junction_id].copy()
            title_suffix = f" - Junction {junction_id}"
        else:
            plot_data = results.copy()
            title_suffix = " - All Junctions"
        if len(plot_data) == 0:
            print("No data to plot")
            return
        if len(plot_data) > max_points:
            plot_data = plot_data.sample(n=max_points).sort_values('datetime')
        plt.figure(figsize=(15, 10))
        if 'actual_vehicles' in plot_data.columns and plot_data['actual_vehicles'].notna().any():
            plt.subplot(2, 2, 1)
            plt.scatter(plot_data['actual_vehicles'], plot_data['predicted_vehicles'], alpha=0.6)
            min_val = min(plot_data['actual_vehicles'].min(), plot_data['predicted_vehicles'].min())
            max_val = max(plot_data['actual_vehicles'].max(), plot_data['predicted_vehicles'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('Actual Vehicles')
            plt.ylabel('Predicted Vehicles')
            plt.title(f'Predictions vs Actual{title_suffix}')
            plt.grid(True, alpha=0.3)
            mae = abs(plot_data['prediction_error']).mean()
            rmse = np.sqrt((plot_data['prediction_error']**2).mean())
            plt.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.subplot(2, 2, 2)
        plt.plot(plot_data['datetime'], plot_data['predicted_vehicles'], label='Predicted', linewidth=2, alpha=0.8)
        if 'actual_vehicles' in plot_data.columns and plot_data['actual_vehicles'].notna().any():
            plt.plot(plot_data['datetime'], plot_data['actual_vehicles'], label='Actual', linewidth=2)
        plt.xlabel('DateTime')
        plt.ylabel('Vehicles')
        plt.title(f'Time Series{title_suffix}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 3)
        plot_data['hour'] = plot_data['datetime'].dt.hour
        hourly_avg = plot_data.groupby('hour')['predicted_vehicles'].mean()
        plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Predicted Vehicles')
        plt.title(f'Average Predictions by Hour{title_suffix}')
        plt.grid(True, alpha=0.3)
        if 'prediction_error' in plot_data.columns and plot_data['prediction_error'].notna().any():
            plt.subplot(2, 2, 4)
            plt.hist(plot_data['prediction_error'], bins=30, edgecolor='black', alpha=0.7)
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title(f'Error Distribution{title_suffix}')
            plt.grid(True, alpha=0.3)
        else:
            plt.subplot(2, 2, 4)
            plt.hist(plot_data['predicted_vehicles'], bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Predicted Vehicles')
            plt.ylabel('Frequency')
            plt.title(f'Prediction Distribution{title_suffix}')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def main():
    print("="*60)
    print("TRAFFIC PREDICTION MODEL - INFERENCE")
    print("="*60)
    import os
    model_file = 'traffic_prediction_cnn_model.h5'
    if not os.path.exists(model_file):
        print(f"\n Model file '{model_file}' not found!")
        print("\nTo fix this:")
        print("1. First run the training script to create and save the model")
        print("2. Make sure the model file is in the same directory")
        print("3. Or specify the correct path to your model file")
        custom_path = input("\nEnter path to your model file (or press Enter to exit): ").strip()
        if custom_path:
            if os.path.exists(custom_path):
                model_file = custom_path
            else:
                print(f"File {custom_path} not found. Exiting.")
                return
        else:
            print("Exiting.")
            return
    print(f"\nInitializing predictor with model: {model_file}")
    predictor = TrafficPredictor(model_file)
    if predictor.model is None:
        print("\n Failed to load model. Cannot proceed.")
        return
    print("\nLoading test data...")
    try:
        test_files = ['traffic_test.csv']
        test_data = None
        for filename in test_files:
            if os.path.exists(filename):
                test_data = pd.read_csv(filename)
                print(f"Test data loaded from {filename}: {test_data.shape}")
                break
        if test_data is None:
            print(" No test file found. Trying manual input...")
            custom_file = input("Enter path to your test CSV file (or press Enter to create sample data): ").strip()
            if custom_file and os.path.exists(custom_file):
                test_data = pd.read_csv(custom_file)
                print(f"Test data loaded from {custom_file}: {test_data.shape}")
            else:
                print("Creating sample test data for demonstration...")
                np.random.seed(42)
                sample_dates = pd.date_range('2024-01-01', periods=200, freq='H')
                test_data = pd.DataFrame({
                    'DateTime': sample_dates,
                    'Junction': np.random.choice([1, 2, 3, 4], 200),
                    'Vehicles': np.random.poisson(20, 200) + np.random.randint(0, 10, 200),
                    'ID': [f"2024010100{i}" for i in range(200)]
                })
                print(" Sample test data created")
        print(f" Data columns: {test_data.columns.tolist()}")
        print(f" Date range: {test_data['DateTime'].min()} to {test_data['DateTime'].max()}")
        print(f" Unique junctions: {sorted(test_data['Junction'].unique())}")
    except Exception as e:
        print(f" Error loading test data: {e}")
        return
    print("\nMaking predictions on test sequences...")
    results = predictor.predict_sequences(test_data)
    print(f"Generated {len(results)} predictions")
    print("\n" + "="*40)
    print("PREDICTION RESULTS SUMMARY")
    print("="*40)
    print(results[['datetime', 'junction', 'predicted_vehicles']].head(10))
    if 'actual_vehicles' in results.columns and results['actual_vehicles'].notna().any():
        print(f"\nPrediction Accuracy Metrics:")
        mae = results['absolute_error'].mean()
        rmse = np.sqrt((results['prediction_error']**2).mean())
        mape = abs(results['percentage_error']).mean()
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Square Error: {rmse:.2f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print("\nGenerating visualizations...")
    predictor.visualize_predictions(results)
    if len(test_data['Junction'].unique()) > 0:
        junction_to_predict = test_data['Junction'].iloc[0]
        print(f"\nPredicting future traffic for Junction {junction_to_predict}...")
        try:
            future_predictions = predictor.predict_future(test_data, junction_to_predict, hours_ahead=12)
            print(f"Next 12 hours predictions for Junction {junction_to_predict}:")
            for i, pred in enumerate(future_predictions):
                print(f"  {pred['datetime'].strftime('%Y-%m-%d %H:%00')}: {pred['predicted_vehicles']:.1f} vehicles")
            plt.figure(figsize=(12, 6))
            future_df = pd.DataFrame(future_predictions)
            plt.plot(future_df['datetime'], future_df['predicted_vehicles'], marker='o', linewidth=2, markersize=6)
            plt.xlabel('DateTime')
            plt.ylabel('Predicted Vehicles')
            plt.title(f'Future Traffic Predictions - Junction {junction_to_predict}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error making future predictions: {e}")
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()
