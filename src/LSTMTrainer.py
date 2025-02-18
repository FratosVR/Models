import tensorflow as tf
import numpy as np
import time
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class LSTMTrainer:
    """LSTM Model training manager.
    """

    def __init__(self):
        """Initializes the LSTMTrainer class.
        """
        # Attributes

        """List of possible values for the dropout value."""
        self.__dropout_list: np.ndarray = np.arange(0, 1, 0.1)
        """List of possible values for the recurrent dropout value."""
        self.__recurrent_dropout_list: np.ndarray = np.arange(0, 1, 0.1)
        """List of possible values for the activation function."""
        self.__activation_list: list[str] = [
            "tanh", "linear", "relu", "sigmoid"]
        """List of possible values for the recurrent activation function."""
        self.__recurrent_activation_list: list[str] = [
            "tanh", "linear", "relu", "sigmoid"]
        """List of possible values for the unroll value."""
        self.__unroll_list: list[bool] = [True, False]
        """List of possible values for the use bias value."""
        self.__use_bias_list: list[bool] = [True, False]

        """List of tensorboard callbacks. Only used for logging the trainings"""
        self.__tensorboard_callbacks: list[tf.keras.callbacks.Tensorboard] = [
            tf.keras.callbacks.TensorBoard(log_dir="./logs/LSTM", histogram_freq=1)]
        """Current dropout value."""
        self.__dropout: float = 0.0
        """Current recurrent dropout value."""
        self.__recurrent_dropout = 0.0
        """Current activation function."""
        self.__activation: str = "tanh"
        """Current recurrent activation function."""
        self.__recurrent_activation: str = "sigmoid"
        """Current unroll value."""
        self.__unroll: bool = False
        """Current use_bias value."""
        self.__use_bias: bool = True
        """Current model."""
        self.__model: tf.keras.Model = None
        """Best score obtained."""
        self.__best_score: float = 0.0
        """Best arguments obtained."""
        self.__best_args: dict = {}
        """Execution time of the best model."""
        self._exec_time: float = 9999999.0

    def __model_generator(self, input_shape: tuple[int, int], output_shape: int):
        """generates a model given an input shape and an output shape.

        Args:
            input_shape (tuple[int, int]): shape of the input
            output_shape (int): categories
        """
        lstm_layer = tf.keras.layers.LSTM(input_shape=input_shape, output_shape=output_shape,
                                          dropout=self.__dropout,
                                          recurrent_dropout=self.__recurrent_dropout,
                                          activation=self.__activation,
                                          recurrent_activation=self.__recurrent_activation,
                                          unroll=self.__unroll,
                                          use_bias=self.__use_bias)
        self.__model = lstm_layer

    def train_with_all_possible_params(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, epochs: int = 10, batch_size: int = 1, num_cats: int = 5):
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
        training_steps: list = list(product(self.__activation_list, self.__dropout_list.tolist(), self.__recurrent_dropout_list.tolist(),
                                            self.__recurrent_activation_list, self.__unroll_list, self.__use_bias_list))

        for ts in training_steps:
            self.__activation = ts[0]
            self.__dropout = ts[1]
            self.__recurrent_dropout = ts[2]
            self.__recurrent_activation = ts[3]
            self.__unroll = ts[6]
            self.__use_bias = ts[7]

            self.__model_generator(X[0].shape, num_cats)

            self.train(X, y, X_val, y_val, epochs, batch_size)

    def __update_best_args(self,  new_accuracy: float, new_time: float):
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

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, X_test: np.ndarray = None, Y_test=np.ndarray, epochs: int = 10, batch_size: int = 1):
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
        if (self.__model is None):
            raise ValueError("Model is null")
        self.__model.compile(optimizer="adam", loss="mse", metrics=[
                             'mae'])

        history = self.__model.fit(x=X, y=y, validation_data=(
            X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=self.__tensorboard_callbacks)

        evaluation = self.__model.evaluate(X_test, Y_test)

        score = evaluation['accuracy']
        time_start = time.time()
        self.__model.predict(X_val[0])
        time_end = time.time()

        if score > self.__best_score:
            self.__update_best_args(score, time_end - time_start)
        elif score == self.__best_score & (time_end - time_start) < self._exec_time:
            self.__update_best_args(score, time_end - time_start)

    def save_model(self):
        """Saves the best model found.
        Raises:
            ValueError: if model is None
        """
        if model is None:
            raise ValueError("Model is null")
        self.__model.save_model("../models/best-lstm.h5")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the category of the given input.

        Args:
            X (np.ndarray): Test input

        Raises:
            ValueError: if model is None

        Returns:
            np.ndarray: predicted category
        """
        if model is None:
            raise ValueError("Model is null")
        return self.__model.predict(X)
