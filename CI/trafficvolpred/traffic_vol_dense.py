"""
Advanced Dense Neural Network model for traffic volume prediction (tabular).
Upgrades from a simple Dense model:
 - Residual Dense blocks
 - Squeeze-and-Excitation (SE) attention block
 - Wide & Deep, Residual-Attention architectures
 - Robust preprocessing (imputation, cyclical features, scaling, outlier clipping)
 - Cosine annealing learning rate scheduler + standard callbacks
 - K-fold cross-validation support
 - Model save/load with preprocessing objects
"""

import os
import json
import pickle
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, Add, Activation, 
    GlobalAveragePooling1D, Reshape, Multiply, Concatenate, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback, LearningRateScheduler
)



import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def mse_rmse_mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def cosine_annealing(epoch, lr_max, lr_min, epochs):
    cos_inner = np.pi * (epoch % epochs) / epochs
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(cos_inner))


def build_cosine_scheduler(lr_max=1e-3, lr_min=1e-6, epochs=100):
    def scheduler(epoch, lr):
        return cosine_annealing(epoch, lr_max, lr_min, epochs)
    return scheduler


def squeeze_excite_block(x, ratio=8):
    init = x
    filters = int(x.shape[-1])
    se = tf.keras.layers.GlobalAveragePooling1D()(tf.expand_dims(x, axis=1)) if len(x.shape) == 2 else tf.keras.layers.GlobalAveragePooling1D()(x)
    se = tf.keras.layers.Reshape((1, filters))(se) if hasattr(se, 'shape') and len(se.shape) == 2 else se
    se = tf.keras.layers.Dense(max(filters // ratio, 4), activation='relu')(se if len(se.shape) > 2 else tf.reshape(se, (-1, filters)))
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    if len(se.shape) > 2:
        se = tf.keras.layers.Reshape((filters,))(se)
    x = tf.keras.layers.Multiply()([init, se])
    return x


def residual_dense_block(x, units, dropout_rate=0.2):
    shortcut = x
    x = Dense(units, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units, activation=None)(x)
    x = BatchNormalization()(x)
    if int(shortcut.shape[-1]) != units:
        shortcut = Dense(units, activation=None)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


class AdvancedDenseTrafficPredictor:
    def __init__(self,
                 model_path: str = "traffic_adv_model",
                 scaler: Optional[StandardScaler] = None,
                 variance_threshold: float = 1e-5,
                 random_state: int = 42):
        self.model: Optional[Model] = None
        self.scaler = scaler
        self.variance_selector: Optional[VarianceThreshold] = None
        self.variance_threshold = variance_threshold
        self.feature_names: List[str] = []
        self.label_encoders = {} 
        self.model_path = model_path
        self.random_state = random_state

    def load_and_preprocess_data(self, file_path: str, dropna: bool = True,
                                 impute_strategy: str = "median",
                                 clip_outliers: bool = True,
                                 outlier_q: float = 0.999) -> pd.DataFrame:
        
        df = pd.read_csv(file_path)
        print(f"[data] loaded {file_path} shape={df.shape}")
        if dropna:
            df = df.dropna(how='all')
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
            df['hour'] = df['date_time'].dt.hour.fillna(0).astype(int)
            df['day_of_week'] = df['date_time'].dt.dayofweek.fillna(0).astype(int)
            df['month'] = df['date_time'].dt.month.fillna(1).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df = df.drop(columns=['date_time'], errors='ignore')

        df = df.drop(columns=['weather_description'], errors='ignore')
        for cat in ['holiday', 'weather_main']:
            if cat in df.columns:
                df[cat] = df[cat].fillna("None")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found after preprocessing. Check your CSV.")
        
        num_df = df[numeric_cols].copy()

        if impute_strategy == "median":
            num_df = num_df.fillna(num_df.median())
        elif impute_strategy == "mean":
            num_df = num_df.fillna(num_df.mean())
        else:
            num_df = num_df.fillna(0)
        if clip_outliers:
            lower_q = (1 - outlier_q) / 2
            upper_q = 1 - lower_q
            for col in num_df.columns:
                low = num_df[col].quantile(lower_q)
                high = num_df[col].quantile(upper_q)
                if low is None or high is None:
                    continue
                num_df[col] = num_df[col].clip(lower=low, upper=high)

                
        other_cols = df.drop(columns=numeric_cols).copy()
        df_processed = pd.concat([other_cols.reset_index(drop=True), num_df.reset_index(drop=True)], axis=1)
        df_processed = df_processed.loc[:, df_processed.columns]
        print(f"[data] processed shape={df_processed.shape}")
        return df_processed

    def prepare_features(self, df: pd.DataFrame, target_col: str = "traffic_volume",
                         do_variance_threshold: bool = True):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].values
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = pd.factorize(X[col])[0]
        if do_variance_threshold:
            if self.variance_selector is None:
                self.variance_selector = VarianceThreshold(self.variance_threshold)
                self.variance_selector.fit(X.values)
            mask = self.variance_selector.get_support()
            X = X.loc[:, mask]
            selected_cols = X.columns.tolist()
        else:
            selected_cols = X.columns.tolist()
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X.values)
        else:
            X_scaled = self.scaler.transform(X.values)
        self.feature_names = selected_cols
        return X_scaled, y

    def build_standard_dense(self, input_dim: int) -> Model:
        inputs = Input(shape=(input_dim,))
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
        return model

    def build_residual_attention_model(self, input_dim: int) -> Model:
        inputs = Input(shape=(input_dim,))
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = residual_dense_block(x, units=512, dropout_rate=0.2)
        x = residual_dense_block(x, units=256, dropout_rate=0.2)
        x = residual_dense_block(x, units=128, dropout_rate=0.15)
        se = Dense(64, activation='relu')(x)
        se = Dense(int(x.shape[-1]), activation='sigmoid')(se)
        x = Multiply()([x, se])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
        return model

    def build_wide_and_deep(self, input_dim: int) -> Model:
        inputs = Input(shape=(input_dim,))
        deep = Dense(256, activation='relu')(inputs)
        deep = BatchNormalization()(deep)
        deep = Dropout(0.25)(deep)
        deep = Dense(128, activation='relu')(deep)
        deep = BatchNormalization()(deep)
        deep = Dropout(0.15)(deep)
        wide = Dense(32, activation='relu')(inputs)
        wide = Dropout(0.1)(wide)
        combined = Concatenate()([wide, deep])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.1)(combined)
        outputs = Dense(1, activation='linear')(combined)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
        return model

    def get_callbacks(self, model_path: str, epochs: int = 100, lr_max: float = 1e-3, lr_min: float = 1e-6):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
            ModelCheckpoint(f'{model_path}_best.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]
        scheduler = LearningRateScheduler(build_cosine_scheduler(lr_max=lr_max, lr_min=lr_min, epochs=epochs))
        callbacks.append(scheduler)
        return callbacks

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              model_type: str = "residual_attention",
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              save_model: bool = True,
              model_name: Optional[str] = None,
              verbose: int = 1) -> Tuple[Any, np.ndarray, np.ndarray]:
        if model_name is None:
            model_name = self.model_path
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split,
                                                          random_state=self.random_state, shuffle=True)
        input_dim = X.shape[1]
        if model_type == 'standard':
            self.model = self.build_standard_dense(input_dim)
        elif model_type == 'wide_deep':
            self.model = self.build_wide_and_deep(input_dim)
        elif model_type == 'residual_attention':
            self.model = self.build_residual_attention_model(input_dim)
        else:
            raise ValueError("model_type must be 'standard', 'residual_attention', or 'wide_deep'")
        print("[train] Model Summary:")
        self.model.summary()
        callbacks = self.get_callbacks(model_name, epochs=epochs, lr_max=1e-3, lr_min=1e-6)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        if save_model:
            self.save(model_name)
        return history, X_val, y_val

    def save(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = self.model_path
        if self.model is None:
            raise ValueError("No model to save.")
        model_file = f"{model_name}.h5"
        self.model.save(model_file)
        print(f"[save] model saved to {model_file}")
        prep = {
            "feature_names": self.feature_names,
            "variance_threshold": self.variance_threshold
        }
        prep_file = f"{model_name}_preprocessing.pkl"
        with open(prep_file, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "variance_selector": self.variance_selector,
                "prep": prep
            }, f)
        print(f"[save] preprocessing saved to {prep_file}")

    def load(self, model_name: Optional[str] = None) -> bool:
        if model_name is None:
            model_name = self.model_path
        model_file = f"{model_name}.h5"
        prep_file = f"{model_name}_preprocessing.pkl"
        try:
            self.model = load_model(model_file, compile=True)
            print(f"[load] model loaded from {model_file}")
        except Exception as e:
            print(f"[load] could not load model: {e}")
            return False
        try:
            with open(prep_file, "rb") as f:
                data = pickle.load(f)
                self.scaler = data.get("scaler")
                self.variance_selector = data.get("variance_selector")
                prep_meta = data.get("prep", {})
                self.feature_names = prep_meta.get("feature_names", self.feature_names)
                self.variance_threshold = prep_meta.get("variance_threshold", self.variance_threshold)
            print(f"[load] preprocessing loaded from {prep_file}")
            return True
        except Exception as e:
            print(f"[load] could not load preprocessing: {e}")
            return False

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("No model loaded.")
        y_pred = self.model.predict(X_val).flatten()
        metrics = mse_rmse_mae_r2(y_val, y_pred)
        metrics['predictions'] = y_pred
        metrics['actual'] = y_val
        print(f"[eval] mse={metrics['mse']:.4f}, rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}, r2={metrics['r2']:.4f}")
        return metrics

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(history.history.get('mae', []), label='train mae')
        plt.plot(history.history.get('val_mae', []), label='val mae')
        plt.xlabel('epoch'); plt.ylabel('mae'); plt.legend(); plt.title('MAE')
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, metrics: Dict[str, Any], n_points: int = 200):
        preds = metrics['predictions']
        actual = metrics['actual']
        n = min(n_points, len(preds))
        plt.figure(figsize=(12, 5))
        plt.plot(range(n), actual[:n], label='actual', marker='o', lw=1)
        plt.plot(range(n), preds[:n], label='predicted', marker='s', lw=1)
        plt.legend(); plt.title('Actual vs Predicted (first {} samples)'.format(n))
        plt.show()

    def approximate_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("No model loaded.")
        first_dense = None
        for layer in self.model.layers:
            if hasattr(layer, "get_weights") and len(layer.get_weights()) > 0:
                w = layer.get_weights()[0]
                if w.ndim == 2:
                    first_dense = layer
                    break
        if first_dense is None:
            raise ValueError("No dense layer with weights found for importance proxy.")
        weights = first_dense.get_weights()[0]
        importance = np.mean(np.abs(weights), axis=1)
        features = self.feature_names if len(self.feature_names) == len(importance) else [f"f{i}" for i in range(len(importance))]
        df = pd.DataFrame({"feature": features, "importance": importance})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True).head(top_n)
        return df

    def explain_with_shap(self, X_sample: np.ndarray, feature_names: Optional[List[str]] = None):
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP package not available. Install shap to use explainability.")
        if self.model is None:
            raise ValueError("No model loaded.")
        if feature_names is None:
            feature_names = self.feature_names
        explainer = shap.KernelExplainer(self.model.predict, X_sample)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        return shap_values

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
            all_feats = list(self.variance_selector.feature_names_in_) if hasattr(self.variance_selector, "feature_names_in_") else self.feature_names
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
            X_ordered = data[self.feature_names].values
            X_scaled = self.scaler.transform(X_ordered)
        return X_scaled

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        X = self.preprocess_new(data)
        preds = self.model.predict(X).flatten()
        return preds

    def predict_batch_from_file(self, file_path: str, output_file: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        Xp = self.preprocess_new(df)
        preds = self.model.predict(Xp).flatten()
        out_df = df.copy()
        out_df['predicted_traffic_volume'] = preds
        if output_file:
            out_df.to_csv(output_file, index=False)
            print(f"[io] predictions written to {output_file}")
        return out_df

    def evaluate_on_file(self, file_path: str, target_col: str = "traffic_volume"):
        df = self.load_and_preprocess_data(file_path)
        X, y = self.prepare_features(df)
        metrics = self.evaluate(X, y)
        comp = pd.DataFrame({"actual": metrics['actual'], "predicted": metrics['predictions']})
        comp['error'] = comp['actual'] - comp['predicted']
        comp['abs_error'] = np.abs(comp['error'])
        comp['pct_error'] = comp['abs_error'] / (comp['actual'] + 1e-9) * 100
        print(comp.head(20).to_string(index=False))
        return comp, metrics

    def cross_validate(self, df: pd.DataFrame, target_col: str = "traffic_volume",
                       model_type: str = "residual_attention", folds: int = 5,
                       epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        X_all, y_all = self.prepare_features(df, target_col=target_col)
        kf = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        fold_results = {}
        fold_idx = 0
        original_model = self.model
        for train_idx, val_idx in kf.split(X_all):
            fold_idx += 1
            print(f"[cv] fold {fold_idx}/{folds}")
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            if model_type == 'standard':
                model = self.build_standard_dense(X_all.shape[1])
            elif model_type == 'wide_deep':
                model = self.build_wide_and_deep(X_all.shape[1])
            else:
                model = self.build_residual_attention_model(X_all.shape[1])
            callbacks = self.get_callbacks(self.model_path + f"_cv_fold{fold_idx}", epochs=epochs, lr_max=1e-3, lr_min=1e-6)
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
            y_pred = model.predict(X_val).flatten()
            metrics = mse_rmse_mae_r2(y_val, y_pred)
            fold_results[f"fold_{fold_idx}"] = metrics
            print(f"  fold {fold_idx} metrics: {metrics}")
            K.clear_session()
        self.model = original_model
        return fold_results

def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Dense Traffic Predictor")
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--file', type=str, default='traffic_data.csv', help='CSV file for training/evaluation')
    parser.add_argument('--model_type', type=str, default='residual_attention', choices=['standard', 'wide_deep', 'residual_attention'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--predict_file', type=str, help='CSV file to generate predictions for')
    parser.add_argument('--load_model', type=str, help='Model name to load')
    args = parser.parse_args()
    predictor = AdvancedDenseTrafficPredictor(model_path='traffic_adv_model')
    if args.load_model:
        predictor.load(args.load_model)
    if args.train:
        df = predictor.load_and_preprocess_data(args.file)
        X, y = predictor.prepare_features(df)
        history, X_val, y_val = predictor.train(X, y, model_type=args.model_type, epochs=args.epochs, batch_size=args.batch_size)
        predictor.plot_training_history(history)
        metrics = predictor.evaluate(X_val, y_val)
        predictor.plot_predictions(metrics)
    if args.predict_file:
        if predictor.model is None:
            loaded = predictor.load()
            if not loaded:
                raise RuntimeError("No model loaded for prediction. Train or provide --load_model.")
        out_df = predictor.predict_batch_from_file(args.predict_file, output_file="predictions_out.csv")
        print(out_df.head())

if __name__ == "__main__":
    predictor = AdvancedDenseTrafficPredictor()
    file_path = "traffic_data.csv"
    df = predictor.load_and_preprocess_data(file_path)
    X, y = predictor.prepare_features(df)
    history, X_val, y_val = predictor.train(
        X, y, model_type="residual_attention", epochs=30, batch_size=32
    )
    results = predictor.evaluate(X_val, y_val)
    predictor.plot_training_history(history)
    predictor.plot_predictions(results)

""" if __name__ == "__main__":
    print("Loading trained model and evaluating on test data...\n")
    predictor = AdvancedDenseTrafficPredictor(model_path="traffic_adv_model")
    if not predictor.load():
        raise RuntimeError("Could not load trained model. Please train it first.")
    test_file = "traffic_test_realistic.csv"
    df_test = predictor.load_and_preprocess_data(test_file)
    X_test, y_test = predictor.prepare_features(df_test)
    results = predictor.evaluate(X_test, y_test)
    predictor.plot_predictions(results)
    print("\nApproximate Feature Importance:")
    importance_df = predictor.approximate_feature_importance(top_n=10)
    print(importance_df.to_string(index=False))
    print("\nPrediction and evaluation completed successfully!")
 """
