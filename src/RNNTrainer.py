import tensorflow as tf
import numpy as np
import time
from itertools import product
import matplotlib.pyplot as plt
import os


class RNNTrainer:
    """RNN Model training manager with hyperparameter tuning using hparams."""

    def __init__(self, interval):
        """Initializes the RNNTrainer class.

        Args:
            interval (float): interval of time between each frame sent to the model.
        """

        # Define hyperparameters for tuning
        """Hparam for activation function."""
        self.HP_ACTIVATION = hp.HParam('activation', hp.Discrete(
            ["tanh", "linear", "relu", "sigmoid"]))
        """Hparam for bias use."""
        self.HP_USE_BIAS = hp.HParam('use_bias', hp.Discrete([True, False]))
        """Hparam for kernel initializer."""
        self.HP_KERNEL_INITIALIZER = hp.HParam(
            'kernel_initializer', hp.Discrete("glorot_uniform", "he_normal", "he_uniform"))
        """Hparam for recurrent initializer."""
        self.HP_RECURRENT_INITIALIZER = hp.HParam('recurrent_initializer', hp.Discrete(
            "glorot_uniform", "he_normal", "he_uniform"))
        """Hparam for bias initializer."""
        self.HP_BIAS_INITIALIZER = hp.HParam('bias_initializer', hp.Discrete(
            "glorot_uniform", "he_normal", "he_uniform"))
        """Hparam for kernel regularizer."""
        self.HP_KERNEL_REGULARIZER = hp.HParam(
            'kernel_regularizer', hp.Discrete("l1", "l2", "l1_l2", None))
        """Hparam for recurrent regularizer."""
        self.HP_RECURRENT_REGULARIZER = hp.HParam(
            'recurrent_regularizer', hp.Discrete("l1", "l2", "l1_l2", None))
        """Hparam for bias regularizer."""
        self.HP_BIAS_REGULARIZER = hp.HParam(
            'bias_regularizer', hp.Discrete("l1", "l2", "l1_l2", None))
        """Hparam for activity regularizer."""
        self.HP_ACTIVITY_REGULARIZER = hp.HParam(
            'activity_regularizer', hp.Discrete("l1", "l2", "l1_l2", None))
        """Hparam for kernel constraint."""
        self.HP_KERNEL_CONSTRAINT = hp.HParam(
            'kernel_constraint', hp.Discrete("max_norm", "non_neg", None))
        """Hparam for recurrent constraint."""
        self.HP_RECURRENT_CONSTRAINT = hp.HParam(
            'recurrent_constraint', hp.Discrete("max_norm", "non_neg", None))
        """Hparam for bias constraint."""
        self.HP_BIAS_CONSTRAINT = hp.HParam(
            'bias_constraint', hp.Discrete("max_norm", "non_neg", None))
        """Hparam for dropout rate."""
        self.HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 1.0))
        """Hparam for recurrent dropout rate."""
        self.HP_RECURRENT_DROPOUT = hp.HParam(
            'recurrent_dropout', hp.RealInterval(0.0, 1.0))
        """Hparam for unroll option."""
        self.HP_UNROLL = hp.HParam('unroll', hp.Discrete([True, False]))
        """Hparam acuracy metric."""
        self.METRIC_ACCURACY = 'accuracy'
        """Interval of time between each frame sent to the model."""
        self.__interval = interval
        """tensorboard log directory."""
        self.__tensorboard_log_dir = f"./logs/hparams_RNN/{interval}"
        self.__tensorboard_callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.__tensorboard_log_dir, histogram_freq=1)
        ]

        # Initialize model parameters
        """Activation function for the current test."""
        self.__activation = "tanh"
        """Use bias option for the current test."""
        self.__use_bias = True
        """Kernel initializer for the current test."""
        self.__kernel_initializer = "glorot_uniform"
        """Recurrent initializer for the current test."""
        self.__recurrent_initializer = "glorot_uniform"
        """Bias initializer for the current test."""
        self.__bias_initializer = "glorot_uniform"
        """Kernel regularizer for the current test."""
        self.__kernel_regularizer = "l1"
        """Recurrent regularizer for the current test."""
        self.__recurrent_regularizer = "l1"
        """Bias regularizer for the current test."""
        self.__bias_regularizer = "l1"
        """Activity regularizer for the current test."""
        self.__activity_regularizer = "l1"
        """Kernel constraint for the current test."""
        self.__kernel_constraint = "max_norm"
        """Recurrent constraint for the current test."""
        self.__recurrent_constraint = "max_norm"
        """Bias constraint for the current test."""
        self.__bias_constraint = "max_norm"
        """Dropout rate for the current test."""
        self.__dropout = 0.0
        """Recurrent dropout rate for the current test."""
        self.__recurrent_dropout = 0.0
        """Unroll option for the current test."""
        self.__unroll = False
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
            tf.keras.layers.RNN(units=output_shape, input_shape=input_shape,
                                dropout=self.__dropout,
                                recurrent_dropout=self.__recurrent_dropout,
                                activation=self.__activation,
                                recurrent_activation=self.__recurrent_activation,
                                unroll=self.__unroll,
                                use_bias=self.__use_bias,
                                kernel_initializer=self.__kernel_initializer,
                                recurrent_initializer=self.__recurrent_initializer,
                                bias_initializer=self.__bias_initializer,
                                kernel_regularizer=self.__kernel_regularizer,
                                recurrent_regularizer=self.__recurrent_regularizer,
                                bias_regularizer=self.__bias_regularizer,
                                activity_regularizer=self.__activity_regularizer,
                                kernel_constraint=self.__kernel_constraint,
                                recurrent_constraint=self.__recurrent_constraint,
                                bias_constraint=self.__bias_constraint
                                ),
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ])

    def train_with_hparams(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, epochs: int = 10, batch_size: int = 1, num_cats: int = 5) -> None:
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

        # Create loggin directory
        os.makedirs(self.__tensorboard_log_dir, exist_ok=True)

        session_num = 0

        # Generate all possible combinations of hyperparameters
        hparams_combinations = list(product(
            self.HP_ACTIVATION.domain.values,
            self.HP_USE_BIAS.domain.values,
            self.HP_KERNEL_INITIALIZER.domain.values,
            self.HP_RECURRENT_INITIALIZER.domain.values,
            self.HP_BIAS_INITIALIZER.domain.values,
            self.HP_KERNEL_REGULARIZER.domain.values,
            self.HP_RECURRENT_REGULARIZER.domain.values,
            self.HP_BIAS_REGULARIZER.domain.values,
            self.HP_ACTIVITY_REGULARIZER.domain.values,
            self.HP_KERNEL_CONSTRAINT.domain.values,
            self.HP_RECURRENT_CONSTRAINT.domain.values,
            self.HP_BIAS_CONSTRAINT.domain.values,
            self.HP_DROPOUT.domain.values,
            self.HP_RECURRENT_DROPOUT.domain.values,
            self.HP_UNROLL.domain.values
        ))

        for hparam_values in hparam_combinations:
            hparams = {
                self.HP_ACTIVATION: hparam_values[0],
                self.HP_USE_BIAS: hparam_values[1],
                self.HP_KERNEL_INITIALIZER: hparam_values[2],
                self.HP_RECURRENT_INITIALIZER: hparam_values[3],
                self.HP_BIAS_INITIALIZER: hparam_values[4],
                self.HP_KERNEL_REGULARIZER: hparam_values[5],
                self.HP_RECURRENT_REGULARIZER: hparam_values[6],
                self.HP_BIAS_REGULARIZER: hparam_values[7],
                self.HP_ACTIVITY_REGULARIZER: hparam_values[8],
                self.HP_KERNEL_CONSTRAINT: hparam_values[9],
                self.HP_RECURRENT_CONSTRAINT: hparam_values[10],
                self.HP_BIAS_CONSTRAINT: hparam_values[11],
                self.HP_DROPOUT: hparam_values[12],
                self.HP_RECURRENT_DROPOUT: hparam_values[13],
                self.HP_UNROLL: hparam_values[14]
            }

            run_name = f"run-{session_num}"
            session_log_dir = os.path.join(
                self.__tensorboard_log_dir, run_name)
            print(f"Starting experiment {run_name} with {hparams}")

            self.__activation = hparam_values[0]
            self.__use_bias = hparam_values[1]
            self.__kernel_initializer = hparam_values[2]
            self.__recurrent_initializer = hparam_values[3]
            self.__bias_initializer = hparam_values[4]
            self.__kernel_regularizer = hparam_values[5]
            self.__recurrent_regularizer = hparam_values[6]
            self.__bias_regularizer = hparam_values[7]
            self.__activity_regularizer = hparam_values[8]
            self.__kernel_constraint = hparam_values[9]
            self.__recurrent_constraint = hparam_values[10]
            self.__bias_constraint = hparam_values[11]
            self.__dropout = hparam_values[12]
            self.__recurrent_dropout = hparam_values[13]
            self.__unroll = hparam_values[14]

            self.__model_generator(X[0].shape, num_cats)
            self.train(X, y, X_val, y_val, epochs,
                       batch_size, session_log_dir, hparams)

            session_num += 1

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, epochs: int = 10, batch_size: int = 1, log_dir: str = None, hparams: dict = None) -> None:
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
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)
            history = self.__model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(
                X_val, y_val), callbacks=self.__tensorboard_callbacks)

            _, accuracy = self.__model.evaluate(X_val, y_val)

            tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)

        time_start: float = time.time()
        self.__model.predict(X_val[0])
        time_end: float = time.time()

        if accuracy > self.__best_score:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
        elif accuracy == self.__best_score and (time_end - time_start) < self._exec_time:
            self.__update_best_args(accuracy, time_end - time_start, hparams)

    def __update_best_args(self, new_accuracy: float, new_time: float, hparams: dict) -> None:
        """updates the best arguments found when a new model is better than the previous one.

        Args:
            new_accuracy (float): new accuracy found
            new_time (float): new execution time found
            hparams (dict): new hyperparameters found
        """
        self.__best_score = new_accuracy
        self._exec_time = new_time
        self.__best_args = hparams

    def save_model(self) -> None:
        """"Saves the best model found.

        Raises:
            ValueError: if the model is None
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        self.__model.save(f"../model/best-rnn{self.__interval}.h5")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the output given an input.

        Args:
            X (np.ndarray): input data

        Returns:
            np.ndarray: output data
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        return self.__model.predict(X)


if __name__ == "__main__":
    X_train = np.random.randn(100, 10, 1)
    y_train = np.random.randn(0, 5, size=(100, 5))

    X_val = np.random.rand(20, 10, 1)
    y_val = np.random.rand(0, 5, size=(20, 5))

    trainer = RNNTrainer(10)
    trainer.train_with_hparams(X_train, y_train, X_val, y_val)

    print(f"Best hyperparameters found: {trainer.__best_args}")
