import tensorflow as tf
import numpy as np
import time
from itertools import product
import os
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp


class LSTMTrainer:
    """LSTM Model training manager with hyperparameter tuning using hparams."""

    def __init__(self):
        """Initializes the LSTMTrainer class."""

        # Define hyperparameters for tuning
        """Hparam for dropout rate."""
        self.HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 1.0))
        """Hparam for recurrent dropout rate."""
        self.HP_RECURRENT_DROPOUT = hp.HParam(
            'recurrent_dropout', hp.RealInterval(0.0, 1.0))
        """Hparam for activation function."""
        self.HP_ACTIVATION = hp.HParam('activation', hp.Discrete(
            ["tanh", "linear", "relu", "sigmoid"]))
        """Hparam for recurrent activation function."""
        self.HP_RECURRENT_ACTIVATION = hp.HParam(
            'recurrent_activation', hp.Discrete(["tanh", "linear", "relu", "sigmoid"]))
        """Hparam for unroll option."""
        self.HP_UNROLL = hp.HParam('unroll', hp.Discrete([True, False]))
        """Hparam for use bias option."""
        self.HP_USE_BIAS = hp.HParam('use_bias', hp.Discrete([True, False]))
        """Hparam acuracy metric."""
        self.METRIC_ACCURACY = 'accuracy'
        """tensorboard log directory."""
        self.__tensorboard_log_dir = "./logs/hparams_LSTM"
        """tensorboard callbacks."""
        self.__tensorboard_callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.__tensorboard_log_dir, histogram_freq=1)
        ]

        # Initialize model parameters
        """Dropout rate for the current test."""
        self.__dropout = 0.0
        """Recurrent dropout rate for the current test."""
        self.__recurrent_dropout = 0.0
        """Activation function for the current test."""
        self.__activation = "tanh"
        """Recurrent activation function for the current test."""
        self.__recurrent_activation = "sigmoid"
        """Unroll option for the current test."""
        self.__unroll = False
        """Use bias option for the current test."""
        self.__use_bias = True
        """Model instance."""
        self.__model = None
        """Best score found."""
        self.__best_score = 0.0
        """Best hyperparameters found."""
        self.__best_args = {}
        """Execution time of the best model."""
        self._exec_time = float('inf')

    def __model_generator(self, input_shape: tuple[int, int], output_shape: int) -> None:
        """generates a model given an input shape and an output shape.

        Args:
            input_shape (tuple[int, int]): shape of the input
            output_shape (int): categories
        """

        self.__model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=output_shape, input_shape=input_shape,
                                 dropout=self.__dropout,
                                 recurrent_dropout=self.__recurrent_dropout,
                                 activation=self.__activation,
                                 recurrent_activation=self.__recurrent_activation,
                                 unroll=self.__unroll,
                                 use_bias=self.__use_bias),
            tf.keras.layers.Dense(output_shape, activation="softmax")
        ])

    def train_with_hparams(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None,
                           epochs: int = 10, batch_size: int = 1, num_cats: int = 5) -> None:
        """Trains a model with all possible parameters.

        Args:
            X (np.ndarray): train X
            y (np.ndarray): train Y
            X_val (np.ndarray, optional): Validation X. Defaults to None.
            y_val (np.ndarray, optional): Validation Y. Defaults to None.
            epochs (int, optional): number of epochs in training. Defaults to 10.
            batch_size (int, optional): batch size in training. Defaults to 1.
            num_cats (int, optional): number of categories to be classified. Defaults to 5.
        """

        log_dir = self.__tensorboard_log_dir

        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)

        hparam_combinations = list(product(
            ["tanh", "linear", "relu", "sigmoid"],  # Activation functions
            np.arange(0, 1, 0.1).tolist(),         # Dropout values
            np.arange(0, 1, 0.1).tolist(),         # Recurrent dropout values
            # Recurrent activation functions
            ["tanh", "linear", "relu", "sigmoid"],
            [True, False],                         # Unroll options
            [True, False]                          # Use bias
        ))

        session_num = 0
        for hparam_values in hparam_combinations:
            hparams = {
                self.HP_ACTIVATION: hparam_values[0],
                self.HP_DROPOUT: hparam_values[1],
                self.HP_RECURRENT_DROPOUT: hparam_values[2],
                self.HP_RECURRENT_ACTIVATION: hparam_values[3],
                self.HP_UNROLL: hparam_values[4],
                self.HP_USE_BIAS: hparam_values[5],
            }

            run_name = f"run-{session_num}"
            session_log_dir = os.path.join(log_dir, run_name)
            print(f"Starting experiment {run_name} with {hparams}")

            self.__activation = hparam_values[0]
            self.__dropout = hparam_values[1]
            self.__recurrent_dropout = hparam_values[2]
            self.__recurrent_activation = hparam_values[3]
            self.__unroll = hparam_values[4]
            self.__use_bias = hparam_values[5]

            self.__model_generator(X[0].shape, num_cats)
            self.train(X, y, X_val, y_val, epochs,
                       batch_size, session_log_dir, hparams)

            session_num += 1

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 1, log_dir: str = None, hparams: dict = None) -> None:
        """Train the model with the given parameters.

        Args:
            X (np.ndarray): train X
            y (np.ndarray): train Y
            X_val (np.ndarray, optional): Validation X. Defaults to None.
            y_val (np.ndarray, optional): Validation Y. Defaults to None.
            epochs (int, optional): number of epochs. Defaults to 10.
            batch_size (int, optional): batch size. Defaults to 1.

        Raises:
            ValueError: if the model is None
        """

        if self.__model is None:
            raise ValueError("Model is not initialized")

        self.__model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

        # Create a writer for TensorBoard logging
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)  # Log hyperparameters

            history = self.__model.fit(X, y, validation_data=(X_val, y_val), epochs=epochs,
                                       batch_size=batch_size, callbacks=self.__tensorboard_callbacks)

            _, accuracy = self.__model.evaluate(X_val, y_val)

            tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)

        time_start = time.time()
        self.__model.predict(X_val[0])
        time_end = time.time()

        # Update best hyperparameters if a better model is found
        if accuracy > self.__best_score:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
        elif accuracy == self.__best_score and (time_end - time_start) < self._exec_time:
            self.__update_best_args(accuracy, time_end - time_start, hparams)

    def __update_best_args(self, new_accuracy: float, new_time: float, hparams: dict) -> None:
        """updates the best arguments found when a new model is better than the previous one.

         Args:
             new_accuracy (float): new accuracy of the model
             new_time (float): new execution time of the model
         """
        self.__dropout = self.__best_args["dropout"]
        self.__recurrent_dropout = self.__best_args["recurrent_dropout"]
        self.__activation = self.__best_args["activation"]
        self.__recurrent_activation = self.__best_args["recurrent_activation"]
        self.__unroll = self.__best_args["unroll"]
        self.__use_bias = self.__best_args["use_bias"]
        self._exec_time = new_time
        self.__best_score = new_accuracy

    def save_model(self) -> None:
        """Saves the best model found.
        Raises:
            ValueError: if model is None
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        self.__model.save("../models/best-lstm.h5")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the category of the given input.

        Args:
            X (np.ndarray): Test input

        Raises:
            ValueError: if model is None

        Returns:
            np.ndarray: predicted category
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        return self.__model.predict(X)


if __name__ == "__main__":
    # Example usage with dummy data
    X_train = np.random.randn(100, 10, 1)
    y_train = np.random.randint(0, 5, size=(100, 5))

    X_val = np.random.randn(20, 10, 1)
    y_val = np.random.randint(0, 5, size=(20, 5))

    trainer = LSTMTrainer()
    trainer.train_with_hparams(
        X_train, y_train, X_val, y_val, epochs=5, batch_size=2)

    print(f"Best hyperparameters: {trainer.__best_args}")
