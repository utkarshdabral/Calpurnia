"""
AI/ML Module: Residual Error Prediction using LSTM/GRU
Trains neural networks to predict the difference between SGP4 predictions and actual positions.
"""

import numpy as np
from typing import Tuple, Optional
import json
from datetime import datetime, timedelta


class ResidualErrorPredictor:
    """
    LSTM-based predictor for orbital propagation residual errors.
    Predicts the difference between where SGP4 says a satellite is vs. where it actually is.
    """
    
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 6):
        """
        Initialize residual error predictor.
        
        Args:
            sequence_length: Number of historical time steps (hours)
            prediction_horizon: Number of time steps to predict ahead (hours)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model_trained = False
        self.feature_scaler = None
        self.target_scaler = None
    
    def create_sequences(
        self,
        sgp4_positions: np.ndarray,
        actual_positions: np.ndarray,
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training from time series data.
        
        Args:
            sgp4_positions: SGP4 predicted positions (N, 3) in km
            actual_positions: Actual positions (N, 3) in km
            sequence_length: Sequence length (uses self.sequence_length if None)
        
        Returns:
            (X_sequences, y_targets) where:
                X_sequences: (M, sequence_length, 3) - input sequences
                y_targets: (M, 3) - target residuals
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Compute residuals
        residuals = actual_positions - sgp4_positions
        
        X_sequences = []
        y_targets = []
        
        for i in range(len(residuals) - sequence_length):
            X_sequences.append(residuals[i:i+sequence_length])
            y_targets.append(residuals[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_targets)
    
    def extract_space_weather_features(
        self,
        timestamps: np.ndarray,
        f107_daily: Optional[np.ndarray] = None,
        ap_index: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract space weather features for enhanced prediction.
        
        Args:
            timestamps: Time stamps (datetime objects or seconds since epoch)
            f107_daily: F10.7 solar flux index (optional)
            ap_index: Ap geomagnetic index (optional)
        
        Returns:
            Features array (N, n_features)
        """
        n_samples = len(timestamps)
        features = []
        
        for i, ts in enumerate(timestamps):
            # Extract time-based features
            if hasattr(ts, 'hour'):
                hour = ts.hour
                day_of_year = ts.timetuple().tm_yday
            else:
                dt = datetime.fromtimestamp(ts)
                hour = dt.hour
                day_of_year = dt.timetuple().tm_yday
            
            feature_vec = [
                hour / 24.0,  # Normalized hour of day (orbital period effect)
                day_of_year / 365.0,  # Normalized day of year (seasonal effect)
            ]
            
            # Add space weather if provided
            if f107_daily is not None:
                feature_vec.append(f107_daily[i] / 200.0)  # Normalized F10.7
            
            if ap_index is not None:
                feature_vec.append(ap_index[i] / 100.0)  # Normalized Ap
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def train_lstm_mock(
        self,
        sgp4_positions: np.ndarray,
        actual_positions: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> dict:
        """
        Mock LSTM training (in production, use TensorFlow/PyTorch).
        Trains model on residual patterns and returns training history.
        
        Args:
            sgp4_positions: SGP4 predictions (N, 3)
            actual_positions: Actual positions (N, 3)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
        
        Returns:
            Training history dictionary
        """
        X, y = self.create_sequences(sgp4_positions, actual_positions)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Compute statistics (mock training)
        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(epochs):
            # Mock loss computation
            train_pred = X_train[:, -1]  # Simple baseline: use last residual
            train_loss = np.mean((train_pred - y_train) ** 2)
            train_loss_history.append(train_loss)
            
            val_pred = X_val[:, -1]
            val_loss = np.mean((val_pred - y_val) ** 2)
            val_loss_history.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
        
        self.model_trained = True
        
        return {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'final_train_loss': float(train_loss_history[-1]),
            'final_val_loss': float(val_loss_history[-1]),
            'n_samples': len(X),
            'n_features': 3,
        }
    
    def predict_residual(
        self,
        recent_residuals: np.ndarray,
        steps_ahead: int = 1
    ) -> np.ndarray:
        """
        Predict residual errors for future time steps.
        
        Args:
            recent_residuals: Recent residual history (sequence_length, 3)
            steps_ahead: Number of steps to predict ahead
        
        Returns:
            Predicted residuals (steps_ahead, 3) in km
        """
        if not self.model_trained:
            # Simple baseline: exponential moving average of residuals
            return np.tile(np.mean(recent_residuals, axis=0), (steps_ahead, 1))
        
        predictions = []
        current_seq = recent_residuals.copy()
        
        for _ in range(steps_ahead):
            # Mock LSTM prediction: use mean with decay
            next_residual = np.mean(current_seq, axis=0) * 0.95  # Decay factor
            predictions.append(next_residual)
            
            # Update sequence
            current_seq = np.vstack([current_seq[1:], [next_residual]])
        
        return np.array(predictions)
    
    def correct_sgp4_prediction(
        self,
        sgp4_position: np.ndarray,
        recent_residuals: np.ndarray
    ) -> np.ndarray:
        """
        Correct SGP4 prediction using predicted residuals.
        
        Args:
            sgp4_position: SGP4 predicted position (3,) in km
            recent_residuals: Recent residual history (sequence_length, 3)
        
        Returns:
            Corrected position (3,) in km
        """
        predicted_residual = self.predict_residual(recent_residuals, steps_ahead=1)[0]
        corrected_position = sgp4_position + predicted_residual
        return corrected_position
    
    def save_model_config(self, filepath: str):
        """Save model configuration to JSON."""
        config = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'model_trained': self.model_trained,
            'timestamp': datetime.utcnow().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_model_config(filepath: str) -> dict:
        """Load model configuration from JSON."""
        with open(filepath, 'r') as f:
            return json.load(f)


class SpaceWeatherDataManager:
    """Manages and provides space weather data for model enhancement."""
    
    @staticmethod
    def get_f107_daily(date: datetime) -> float:
        """
        Get F10.7 solar flux for given date.
        In production, query from NOAA or other data sources.
        
        Args:
            date: Date
        
        Returns:
            F10.7 index value (typically 70-300 SFU)
        """
        # Mock data: cycle with period of ~11 years (11-year solar cycle)
        day_of_cycle = (date.toordinal() % (11 * 365))
        f107 = 100 + 50 * np.sin(2 * np.pi * day_of_cycle / (11 * 365))
        return f107
    
    @staticmethod
    def get_ap_index(date: datetime) -> float:
        """
        Get Ap geomagnetic index for given date.
        
        Args:
            date: Date
        
        Returns:
            Ap index (typically 0-400)
        """
        # Mock data: random geomagnetic activity
        np.random.seed(date.toordinal())
        ap = 20 + 50 * np.random.random()
        return ap
    
    @staticmethod
    def compute_atmospheric_density_factor(f107: float, ap: float) -> float:
        """
        Compute atmospheric density factor for drag calculations.
        Higher values = higher density = more drag.
        
        Args:
            f107: F10.7 solar flux
            ap: Ap geomagnetic index
        
        Returns:
            Density factor (0.5 to 2.0 range typical)
        """
        # Simplified model
        f107_factor = (f107 / 100.0) * 0.8
        ap_factor = (ap / 50.0) * 0.2
        density_factor = 0.5 + f107_factor + ap_factor
        return min(2.0, density_factor)  # Cap at 2.0
