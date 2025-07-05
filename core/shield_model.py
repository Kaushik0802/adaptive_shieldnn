# core/shield_model.py

import os
import tensorflow as tf
import numpy as np

class ShieldModel:
    """
    Wrapper for the fixed-radius safety control-filter networks (δ filters).
    These are pre-trained from Yasser's MATLAB code and exported via Keras.
    """

    def __init__(self, model_path: str = "3.h5"):
        """
        Initialize the ShieldNN model.
        Args:
            model_path (str): Path to the trained .h5 model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """
        Predicts the safe control output using the safety NN.

        Args:
            x_input (np.ndarray): Input array of shape (N, 2) → [xi, beta]

        Returns:
            np.ndarray: Predicted values (N, 1)
        """
        if len(x_input.shape) == 1:
            x_input = np.expand_dims(x_input, axis=0)
        return self.model.predict(x_input, verbose=0)

    def __call__(self, x_input: np.ndarray) -> np.ndarray:
        return self.predict(x_input)
