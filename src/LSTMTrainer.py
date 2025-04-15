import tensorflow as tf
import numpy as np
import time
from itertools import product
import os
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
from tensorboard import program


class LSTMTrainer:
    """LSTM Model training manager with hyperparameter tuning using hparams."""

    def __init__(self, interval: float):
        """Initializes the LSTMTrainer class."""
        self.HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 1.0))
        self.HP_RECURRENT_DROPOUT = hp.HParam(
            'recurrent_dropout', hp.RealInterval(0.0, 1.0))
        self.HP_ACTIVATION = hp.HParam('activation', hp.Discrete(
            ["tanh", "linear", "relu", "sigmoid"]))
        self.HP_RECURRENT_ACTIVATION = hp.HParam(
            'recurrent_activation', hp.Discrete(["tanh", "linear", "relu", "sigmoid"]))
        self.HP_UNROLL = hp.HParam('unroll', hp.Discrete([True, False]))
        self.HP_USE_BIAS = hp.HParam('use_bias', hp.Discrete([True, False]))
        self.METRIC_ACCURACY = 'accuracy'
        self.__interval = interval
        self.__tensorboard_log_dir = f"./logs/hparams_LSTM{interval}"
        self.__tensorboard_callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.__tensorboard_log_dir, histogram_freq=1)
        ]

        self._dropout_values = [0.0, 0.2, 0.5]
        self._recurrent_dropout_values = [0.0, 0.2, 0.5]

        self.__dropout = 0.0
        self.__recurrent_dropout = 0.0
        self.__activation = "tanh"
        self.__recurrent_activation = "sigmoid"
        self.__unroll = False
        self.__use_bias = True
        self.__model = None
        self.__best_score = 0.0
        self.__best_args = {}
        self._exec_time = float('inf')
        self.__best_model_path = None

    def __model_generator(self, input_shape: tuple[int, int], output_shape: int) -> None:
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
                           epochs: int = 10, batch_size: int = 1, num_cats: int = 6) -> None:
        os.makedirs(self.__tensorboard_log_dir, exist_ok=True)
        session_num = 0

        hparam_combinations = list(product(
            self.HP_ACTIVATION.domain.values,
            self._dropout_values,
            self._recurrent_dropout_values,
            self.HP_RECURRENT_ACTIVATION.domain.values,
            self.HP_UNROLL.domain.values,
            self.HP_USE_BIAS.domain.values
        ))

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
            session_log_dir = os.path.join(
                self.__tensorboard_log_dir, run_name)
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

        if self.__model is None:
            raise ValueError("Model is not initialized")

        self.__model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)

            history = self.__model.fit(X, y, validation_data=(X_val, y_val), epochs=epochs,
                                       batch_size=batch_size, callbacks=self.__tensorboard_callbacks)

            _, accuracy = self.__model.evaluate(X_val, y_val)
            tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)

        time_start = time.time()
        self.__model.predict(X_val[0:1])
        time_end = time.time()

        if accuracy > self.__best_score:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
        elif accuracy == self.__best_score and (time_end - time_start) < self._exec_time:
            self.__update_best_args(accuracy, time_end - time_start, hparams)

    def __update_best_args(self, new_accuracy: float, new_time: float, hparams: dict) -> None:
        self.__best_args = hparams
        self._exec_time = new_time
        self.__best_score = new_accuracy

    def save_model(self) -> None:
        if self.__model is None:
            raise ValueError("Model is not initialized")
        self.__model.save(f"../models/best-lstm{self.__interval}.h5")
        self.__best_model_path = f"../models/best-lstm{self.__interval}.h5"

    def best_model(self):
        return self.__best_model_path

    def stats(self) -> str:
        return f"Best score: {self.__best_score}, Best args: {self.__best_args}, Execution time: {self._exec_time}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.__model is None:
            raise ValueError("Model is not initialized")
        return self.__model.predict(X)

    def get_log_dir(self) -> str:
        return self.__tensorboard_log_dir


if __name__ == "__main__":
    X_train = np.random.randn(100, 10, 1)
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, 5, size=(100,)), num_classes=5)

    X_val = np.random.randn(20, 10, 1)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 5, size=(20,)), num_classes=5)

    trainer = LSTMTrainer(10)
    trainer.train_with_hparams(
        X_train, y_train, X_val, y_val, epochs=5, batch_size=2)

    print(trainer.stats())

    # Start TensorBoard
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", trainer.get_log_dir()])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
