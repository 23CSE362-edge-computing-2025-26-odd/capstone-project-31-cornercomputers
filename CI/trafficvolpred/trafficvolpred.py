import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

class TrafficVolumeCNN:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        
    def load_and_preprocess_data(self, file_path):
        print("Loading data...")
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        df = df.dropna()
        
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
            df['hour'] = df['date_time'].dt.hour
            df['day_of_week'] = df['date_time'].dt.dayofweek
            df['month'] = df['date_time'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df = df.drop('date_time', axis=1)
        
        categorical_cols = ['holiday', 'weather_main']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                df = df.drop(col, axis=1)
        
        if 'weather_description' in df.columns:
            df = df.drop('weather_description', axis=1)
        
        print(f"Processed data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def prepare_features(self, df):
        target_col = 'traffic_volume'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        self.feature_names = feature_cols
        return X_reshaped, y
    

    
    def build_cnn_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=input_shape, padding='same'),
            BatchNormalization(),
            
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(128, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32, save_model=True, model_path='traffic_cnn_model'):
        print("Building CNN model...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        self.model = self.build_cnn_model((X.shape[1], X.shape[2]))
        
        print("\nModel Architecture:")
        self.model.summary()

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        if save_model:
            checkpoint = ModelCheckpoint(
                f'{model_path}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        

        if save_model:
            self.save_model(model_path)
        
        return history, X_val, y_val
    
    def save_model(self, model_path='traffic_cnn_model'):
        if self.model is None:
            raise ValueError("No model to save! Train the model first.")
        
        model_file = f'{model_path}.h5'
        self.model.save(model_file)
        print(f"Model saved as: {model_file}")
        
        preprocessing_file = f'{model_path}_preprocessing.pkl'
        preprocessing_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        with open(preprocessing_file, 'wb') as f:
            pickle.dump(preprocessing_data, f)
        print(f"Preprocessing objects saved as: {preprocessing_file}")
        
        print(f"\nTo load your model later, use:")
        print(f"model = TrafficVolumeCNN()")
        print(f"model.load_model('{model_path}')")
    
    def load_model(self, model_path='traffic_cnn_model'):
        model_file = f'{model_path}.h5'
        try:
            self.model = load_model(model_file)
            print(f"Model loaded from: {model_file}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
        preprocessing_file = f'{model_path}_preprocessing.pkl'
        try:
            with open(preprocessing_file, 'rb') as f:
                preprocessing_data = pickle.load(f)
            
            self.scaler = preprocessing_data['scaler']
            self.label_encoders = preprocessing_data['label_encoders']
            self.feature_names = preprocessing_data['feature_names']
            print(f"Preprocessing objects loaded from: {preprocessing_file}")
            return True
        except Exception as e:
            print(f"Error loading preprocessing objects: {e}")
            return False
    
    def evaluate_model(self, X_val, y_val):
        print("\nEvaluating model...")
        
        y_pred = self.model.predict(X_val)
        y_pred = y_pred.flatten()
        
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"Validation Metrics:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_val
        }
    
    def plot_training_history(self, history):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        if 'lr' in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, results):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(results['actual'], results['predictions'], alpha=0.6)
        plt.plot([results['actual'].min(), results['actual'].max()], 
                [results['actual'].min(), results['actual'].max()], 'r--', lw=2)
        plt.xlabel('Actual Traffic Volume')
        plt.ylabel('Predicted Traffic Volume')
        plt.title('Actual vs Predicted Traffic Volume')
        plt.text(0.05, 0.95, f"R² = {results['r2']:.4f}", transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.subplot(1, 2, 2)
        residuals = results['actual'] - results['predictions']
        plt.scatter(results['predictions'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Traffic Volume')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_data(self, new_data):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        processed_data = self.preprocess_new_data(new_data)

        prediction = self.model.predict(processed_data)
        return prediction.flatten()[0]
    
    def preprocess_new_data(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
            df['hour'] = df['date_time'].dt.hour
            df['day_of_week'] = df['date_time'].dt.dayofweek
            df['month'] = df['date_time'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df = df.drop('date_time', axis=1)
        
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna('None')
                df[col + '_encoded'] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
                df = df.drop(col, axis=1)
        
        if 'weather_description' in df.columns:
            df = df.drop('weather_description', axis=1)
        
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        df = df[self.feature_names]

        X = self.scaler.transform(df.values)
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X_reshaped

def load_trained_model(model_path='traffic_cnn_model'):
    model = TrafficVolumeCNN()
    if model.load_model(model_path):
        return model
    else:
        return None

def train_new_model():
    cnn_predictor = TrafficVolumeCNN()
    
    file_path = 'traffic_data.csv' 
    df = cnn_predictor.load_and_preprocess_data(file_path)
    
    X, y = cnn_predictor.prepare_features(df)
    
    history, X_val, y_val = cnn_predictor.train_model(X, y, epochs=100)

    results = cnn_predictor.evaluate_model(X_val, y_val)

    cnn_predictor.plot_training_history(history)
    cnn_predictor.plot_predictions(results)
    
    return cnn_predictor

def make_prediction_with_saved_model(new_data, model_path='traffic_cnn_model'):
    model = load_trained_model(model_path)
    
    if model is None:
        print("Could not load model. Please train a new model first.")
        return None

    try:
        prediction = model.predict_new_data(new_data)
        print(f"Predicted traffic volume: {prediction:.0f}")
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def evaluate_saved_model_on_test_data(test_file_path, model_path='traffic_cnn_model'):
    model = load_trained_model(model_path)
    
    if model is None:
        print("Could not load model. Please train a new model first.")
        return None, None
    
    try:
        print(f"Loading test data from: {test_file_path}")
        df = model.load_and_preprocess_data(test_file_path)
        
        target_col = 'traffic_volume'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].values
        y_actual = df[target_col].values

        X_scaled = model.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        y_pred = model.model.predict(X_reshaped).flatten()
        
        results = model.evaluate_model(X_reshaped, y_actual)
        
        comparison_df = pd.DataFrame({
            'Actual': y_actual,
            'Predicted': y_pred,
            'Difference': y_actual - y_pred,
            'Absolute_Error': np.abs(y_actual - y_pred),
            'Percentage_Error': np.abs((y_actual - y_pred) / y_actual) * 100
        })
        
        print("\n" + "="*60)
        print("ACTUAL vs PREDICTED COMPARISON")
        print("="*60)
        
        print("\nFirst 20 predictions:")
        print(comparison_df.head(20).to_string(index=False))
        
        print(f"\n\nSUMMARY STATISTICS:")
        print(f"Total samples: {len(y_actual)}")
        print(f"Mean Actual: {y_actual.mean():.2f}")
        print(f"Mean Predicted: {y_pred.mean():.2f}")
        print(f"Mean Absolute Error: {comparison_df['Absolute_Error'].mean():.2f}")
        print(f"Mean Percentage Error: {comparison_df['Percentage_Error'].mean():.2f}%")
        print(f"Max Error: {comparison_df['Absolute_Error'].max():.2f}")
        print(f"Min Error: {comparison_df['Absolute_Error'].min():.2f}")
        print(f"\nWORST 10 PREDICTIONS (Highest Errors):")
        worst_predictions = comparison_df.nlargest(10, 'Absolute_Error')
        print(worst_predictions.to_string(index=False))
        print(f"\nBEST 10 PREDICTIONS (Lowest Errors):")
        best_predictions = comparison_df.nsmallest(10, 'Absolute_Error')
        print(best_predictions.to_string(index=False))

        model.plot_predictions(results)

        plot_detailed_comparison(comparison_df)
        
        return comparison_df, results
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None, None

def plot_detailed_comparison(comparison_df):
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Actual vs Predicted scatter
    plt.subplot(2, 3, 1)
    plt.scatter(comparison_df['Actual'], comparison_df['Predicted'], alpha=0.6)
    plt.plot([comparison_df['Actual'].min(), comparison_df['Actual'].max()], 
             [comparison_df['Actual'].min(), comparison_df['Actual'].max()], 'r--', lw=2)
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.title('Actual vs Predicted')
    
    # Plot 2: Error distribution
    plt.subplot(2, 3, 2)
    plt.hist(comparison_df['Difference'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.axvline(x=0, color='r', linestyle='--')
    
    # Plot 3: Absolute error distribution
    plt.subplot(2, 3, 3)
    plt.hist(comparison_df['Absolute_Error'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Absolute Error Distribution')
    
    # Plot 4: Time series comparison (first 100 points)
    plt.subplot(2, 3, 4)
    n_points = min(100, len(comparison_df))
    x_range = range(n_points)
    plt.plot(x_range, comparison_df['Actual'][:n_points], label='Actual', marker='o', markersize=3)
    plt.plot(x_range, comparison_df['Predicted'][:n_points], label='Predicted', marker='s', markersize=3)
    plt.xlabel('Sample Index')
    plt.ylabel('Traffic Volume')
    plt.title(f'Actual vs Predicted (First {n_points} samples)')
    plt.legend()
    
    # Plot 5: Percentage error distribution
    plt.subplot(2, 3, 5)
    plt.hist(comparison_df['Percentage_Error'], bins=50, alpha=0.7, edgecolor='black', color='green')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('Percentage Error Distribution')
    
    # Plot 6: Error vs Predicted values
    plt.subplot(2, 3, 6)
    plt.scatter(comparison_df['Predicted'], comparison_df['Difference'], alpha=0.6)
    plt.xlabel('Predicted Traffic Volume')
    plt.ylabel('Error (Actual - Predicted)')
    plt.title('Error vs Predicted Values')
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.show()

def compare_predictions_interactive(model_path='traffic_cnn_model'):
    """Interactive function to compare actual vs predicted values"""
    print("\nChoose how to check actual vs predicted:")
    print("1. Use test data file")
    print("2. Use validation data from training")
    print("3. Manual data entry")
    
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == '1':
        test_file = input("Enter test data file path: ")
        return evaluate_saved_model_on_test_data(test_file, model_path)
        
    elif choice == '2':
        print("This requires reloading and splitting your training data...")
        train_file = input("Enter training data file path: ")

        model = load_trained_model(model_path)
        if model is None:
            return None, None

        df = model.load_and_preprocess_data(train_file)
        X, y = model.prepare_features(df)

        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        results = model.evaluate_model(X_val, y_val)

        comparison_df = pd.DataFrame({
            'Actual': y_val,
            'Predicted': results['predictions'],
            'Difference': y_val - results['predictions'],
            'Absolute_Error': np.abs(y_val - results['predictions']),
            'Percentage_Error': np.abs((y_val - results['predictions']) / y_val) * 100
        })
        
        print("\nValidation Set Comparison:")
        print(comparison_df.head(20).to_string(index=False))
        
        model.plot_predictions(results)
        plot_detailed_comparison(comparison_df)
        
        return comparison_df, results
        
    elif choice == '3':
        model = load_trained_model(model_path)
        if model is None:
            return None, None
            
        print("Enter data for prediction (press Enter for default values):")

        holiday = input("Holiday (None/New Year/Christmas): ") or None
        temp = float(input("Temperature (e.g., 290.5): ") or "290.5")
        rain_1h = float(input("Rain 1h (e.g., 0.0): ") or "0.0")
        snow_1h = float(input("Snow 1h (e.g., 0.0): ") or "0.0")
        clouds_all = int(input("Clouds all (0-100, e.g., 60): ") or "60")
        weather_main = input("Weather main (Clear/Rain/Snow/Clouds): ") or "Clear"
        date_time = input("Date time (YYYY-MM-DD HH:MM:SS): ") or "2024-03-15 14:00:00"
        actual_volume = float(input("Actual traffic volume: "))
        
        new_data = {
            'holiday': holiday,
            'temp': temp,
            'rain_1h': rain_1h,
            'snow_1h': snow_1h,
            'clouds_all': clouds_all,
            'weather_main': weather_main,
            'date_time': date_time
        }
        
        prediction = model.predict_new_data(new_data)
        
        print(f"\n{'='*50}")
        print("SINGLE PREDICTION COMPARISON")
        print(f"{'='*50}")
        print(f"Actual Traffic Volume:    {actual_volume:.0f}")
        print(f"Predicted Traffic Volume: {prediction:.0f}")
        print(f"Difference:               {actual_volume - prediction:.0f}")
        print(f"Absolute Error:           {abs(actual_volume - prediction):.0f}")
        print(f"Percentage Error:         {abs((actual_volume - prediction) / actual_volume) * 100:.2f}%")
        
        return None, None
        
    else:
        print("Invalid choice!")
        return None, None

def main():
    print("Traffic Volume CNN Predictor")
    print("1. Train new model")
    print("2. Load existing model and make prediction")
    print("3. Train model and make prediction")
    print("4. Compare actual vs predicted values")
    
    choice = input("Enter your choice (1, 2, 3, or 4): ")
    
    if choice == '1':
        model = train_new_model()
        return model
        
    elif choice == '2':
        model_path = input("Enter model path (or press Enter for default 'traffic_cnn_model'): ")
        if not model_path:
            model_path = 'traffic_cnn_model'
        
        new_data = {
            'holiday': None,
            'temp': 290.5,
            'rain_1h': 0.2,
            'snow_1h': 0,
            'clouds_all': 60,
            'weather_main': 'Rain',
            'weather_description': 'light rain',
            'date_time': '2024-03-15 14:00:00'
        }
        
        prediction = make_prediction_with_saved_model(new_data, model_path)
        return prediction
        
    elif choice == '3':
        model = train_new_model()
        
        new_data = {
            'holiday': None,
            'temp': 290.5,
            'rain_1h': 0.2,
            'snow_1h': 0,
            'clouds_all': 60,
            'weather_main': 'Rain',
            'weather_description': 'light rain',
            'date_time': '2024-03-15 14:00:00'
        }
        
        try:
            prediction = model.predict_new_data(new_data)
            print(f"\nPredicted traffic volume: {prediction:.0f}")
        except Exception as e:
            print(f"Error making prediction: {e}")
        
        return model
    
    elif choice == '4':
        model_path = input("Enter model path (or press Enter for default 'traffic_cnn_model'): ")
        if not model_path:
            model_path = 'traffic_cnn_model'
            
        comparison_df, results = compare_predictions_interactive(model_path)
        return comparison_df, results
    
    else:
        print("Invalid choice!")
        return None

def quick_train():
    return train_new_model()

def quick_predict(new_data=None, model_path='traffic_cnn_model'):
    if new_data is None:
        new_data = {
            'holiday': None,
            'temp': 290.5,
            'rain_1h': 0.2,
            'snow_1h': 0,
            'clouds_all': 60,
            'weather_main': 'Rain',
            'weather_description': 'light rain',
            'date_time': '2024-03-15 14:00:00'
        }
    
    return make_prediction_with_saved_model(new_data, model_path)

def quick_compare(test_file='metro_data.csv', model_path='traffic_cnn_model'):
    return evaluate_saved_model_on_test_data(test_file, model_path)

def batch_predict(data_file, model_path='traffic_cnn_model', output_file='predictions.csv'):
    model = load_trained_model(model_path)
    
    if model is None:
        print("Could not load model. Please train a new model first.")
        return None
    
    try:
        print(f"Loading data from: {data_file}")
        df = pd.read_csv(data_file)
        original_df = df.copy()
        
        processed_df = model.load_and_preprocess_data(data_file)
        
        has_target = 'traffic_volume' in processed_df.columns
        
        if has_target:
            target_col = 'traffic_volume'
            feature_cols = [col for col in processed_df.columns if col != target_col]
            X = processed_df[feature_cols].values
            y_actual = processed_df[target_col].values
        else:
            feature_cols = list(processed_df.columns)
            X = processed_df[feature_cols].values
            y_actual = None
        
        X_scaled = model.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        y_pred = model.model.predict(X_reshaped).flatten()

        results_df = original_df.copy()
        results_df['Predicted_Traffic_Volume'] = y_pred
        
        if has_target:
            results_df['Actual_Traffic_Volume'] = y_actual
            results_df['Difference'] = y_actual - y_pred
            results_df['Absolute_Error'] = np.abs(y_actual - y_pred)
            results_df['Percentage_Error'] = np.abs((y_actual - y_pred) / y_actual) * 100

            print(f"\nBatch Prediction Summary:")
            print(f"Total predictions: {len(y_pred)}")
            print(f"Mean Predicted: {y_pred.mean():.2f}")
            if y_actual is not None:
                print(f"Mean Actual: {y_actual.mean():.2f}")
                print(f"Mean Absolute Error: {np.abs(y_actual - y_pred).mean():.2f}")
                print(f"Mean Percentage Error: {(np.abs((y_actual - y_pred) / y_actual) * 100).mean():.2f}%")

        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"Error making batch predictions: {e}")
        return None

def get_feature_importance(model_path='traffic_cnn_model'):
    model = load_trained_model(model_path)
    
    if model is None:
        print("Could not load model.")
        return None
    
    print("Feature names in the model:")
    for i, feature in enumerate(model.feature_names):
        print(f"{i+1}. {feature}")
    
    return model.feature_names

def model_summary(model_path='traffic_cnn_model'):
    model = load_trained_model(model_path)
    
    if model is None:
        print("Could not load model.")
        return None
    
    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    print(f"\nFeature Names ({len(model.feature_names)}):")
    for i, feature in enumerate(model.feature_names, 1):
        print(f"  {i}. {feature}")
    
    print(f"\nLabel Encoders:")
    for col, encoder in model.label_encoders.items():
        print(f"  {col}: {list(encoder.classes_)}")
    
    print(f"\nModel Architecture:")
    if model.model:
        model.model.summary()
    else:
        print("  No model loaded")
    
    return {
        'feature_names': model.feature_names,
        'label_encoders': model.label_encoders,
        'model': model.model
    }

def create_sample_prediction_data():
    sample_data = [
        {
            'holiday': None,
            'temp': 290.5,
            'rain_1h': 0.0,
            'snow_1h': 0.0,
            'clouds_all': 20,
            'weather_main': 'Clear',
            'date_time': '2024-03-15 08:00:00'  
        },
        {
            'holiday': None,
            'temp': 285.2,
            'rain_1h': 2.5,
            'snow_1h': 0.0,
            'clouds_all': 80,
            'weather_main': 'Rain',
            'date_time': '2024-03-15 17:30:00'
        },
        {
            'holiday': None,
            'temp': 295.0,
            'rain_1h': 0.0,
            'snow_1h': 0.0,
            'clouds_all': 10,
            'weather_main': 'Clear',
            'date_time': '2024-03-16 14:00:00'
        },
        {
            'holiday': 'Christmas',
            'temp': 280.0,
            'rain_1h': 0.0,
            'snow_1h': 5.0,
            'clouds_all': 90,
            'weather_main': 'Snow',
            'date_time': '2024-12-25 10:00:00'
        }
    ]
    
    return sample_data

def test_model_with_samples(model_path='traffic_cnn_model'):
    model = load_trained_model(model_path)
    
    if model is None:
        print("Could not load model.")
        return None
    
    sample_data = create_sample_prediction_data()
    
    print("="*60)
    print("TESTING MODEL WITH SAMPLE DATA")
    print("="*60)
    
    results = []
    for i, data in enumerate(sample_data, 1):
        try:
            prediction = model.predict_new_data(data)
            
            print(f"\nSample {i}:")
            print(f"  Date/Time: {data['date_time']}")
            print(f"  Weather: {data['weather_main']}")
            print(f"  Temperature: {data['temp']} K")
            print(f"  Holiday: {data['holiday']}")
            print(f"  Predicted Traffic Volume: {prediction:.0f}")
            
            results.append({
                'sample': i,
                'date_time': data['date_time'],
                'weather': data['weather_main'],
                'holiday': data['holiday'],
                'prediction': prediction
            })
            
        except Exception as e:
            print(f"Error with sample {i}: {e}")
    
    return results


def validate_data_format(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data file: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        required_cols = ['temp', 'clouds_all']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
        else:
            print("✓ Data format looks good!")
        
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    print("Traffic Volume CNN Predictor")
    main()