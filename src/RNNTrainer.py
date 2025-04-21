import tensorflow as tf
import numpy as np
import time
from itertools import product
import os
import json
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
from tensorboard import program
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from Utils import plot_confusion_matrix
from keras_tuner import Hyperband, HyperParameters


class RNNTrainer:
    """RNN Model training manager with hyperparameter tuning using hparams."""

    def __init__(self, interval: str, tensorboard_log_dir: str = "./logs/hparams_RNN"):
        """Initializes the RNNTrainer class.

        Args:
            interval (float): interval of time between each frame sent to the model.
        """

        self.__interval = interval
        self.__model = None
        self.__best_model_path = f"best_rnn_{interval}.keras"
        self.__tensorboard_log_dir = tensorboard_log_dir
        self.__tensorboard_callbacks = [TensorBoard(
            log_dir=os.path.join(tensorboard_log_dir, interval))]
        self.__best_acc = 0.0
        self.__tuner = None

    def __model_generator(self, input_shape: tuple[int, int], output_shape: int) -> None:
        """LEGACY

        Generates a model given an input shape and an output shape.

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

    def train_with_hparams(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None,
                           epochs: int = 10, batch_size: int = 1, num_cats: int = 6, categories: list[str] = None) -> None:
        """Train the model with all the combinations of Hparams.

        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Input categories
            X_val (np.ndarray, optional): Validation input data. Defaults to None.
            y_val (np.ndarray, optional): Validation categories data. Defaults to None.
            epochs (int, optional): Number of epochs. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 1.
            num_cats (int, optional): Number of categories. Defaults to 6.
            categories (list[str], optional): List of categories. Defaults to None.
        """

        self.__input_shape = X.shape[1:]
        self.__num_cats = num_cats

        tuner_logdir = os.path.join(self.__tensorboard_log_dir, "keras_tuner")
        os.makedirs(tuner_logdir, exist_ok=True)

        tuner = Hyperband(
            self.__build_model,
            objective="val_accuracy",
            max_epochs=epochs,
            factor=3,
            directory=tuner_logdir,
            project_name=f"tune_lstm_{self.__interval}",
            overwrite=True
        )
        self.__tuner = tuner

        tuner.search(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.__tensorboard_callbacks + [
                EarlyStopping(monitor="val_accuracy", patience=3)
            ],
            verbose=1
        )

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(1)[0]

        self.__model = best_model
        self.__update_best_args(best_model.evaluate(X_val, y_val)[
                                1], best_hp.values)
        self.save_model()

        self.confusion_matrix(
            self.__best_model_path,
            y_true=np.concatenate((y, y_val)),
            y_pred=np.concatenate((self.predict(X), self.predict(X_val))),
            tags=categories
        )

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 1, log_dir: str = None, hparams: dict = None) -> None:
        """Train the model with the given parameters.

        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Input categories
            X_val (np.ndarray, optional): Validation input data. Defaults to None.
            y_val (np.ndarray, optional): Validation categories data. Defaults to None.
            epochs (int, optional): Number of epochs. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 1.
            log_dir (str, optional): Log directory. Defaults to None.
            hparams (dict, optional): Hyperparameters. Defaults to None.
        """
        input_shape = X.shape[1:]

        self.__model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=num_cats, input_shape=input_shape),
            tf.keras.layers.Dense(num_cats, activation="softmax")
        ])

        self.__model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = self.__model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.__tensorboard_callbacks
        )

        acc = history.history["val_accuracy"][-1]
        self.__update_best_args(acc)
        self.save_model()

        self.confusion_matrix(
            self.__best_model_path,
            y_true=np.concatenate((y, y_val)),
            y_pred=np.concatenate((self.predict(X), self.predict(X_val))),
            tags=categories
        )

    def __build_model(self, hp: HyperParameters):
        """Build the model with the given hyperparameters.
        Args:
            hp (HyperParameters): Hyperparameters (automatically created by keras tuner)
        Returns:
            tf.keras.Model: Model
        """
        l1 = hp.Float("l1", 0.0, 0.01, step=0.001)
        l2 = hp.Float("l2", 0.0, 0.01, step=0.001)

        def get_regularizer(name):
            if name == "l1":
                return tf.keras.regularizers.l1(l1)
            elif name == "l2":
                return tf.keras.regularizers.l2(l2)
            elif name == "l1_l2":
                return tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
            return None

        def get_constraint(name):
            if name == "max_norm":
                return tf.keras.constraints.max_norm()
            elif name == "non_neg":
                return tf.keras.constraints.non_neg()
            elif name == "unit_norm":
                return tf.keras.constraints.unit_norm()
            return None

        self.__dropout = hp.Float("dropout", 0.0, 1.0)
        self.__recurrent_dropout = hp.Float("recurrent_dropout", 0.0, 1.0)
        self.__activation = hp.Choice(
            "activation", ["tanh", "linear", "relu", "sigmoid"])
        self.__unroll = hp.Boolean("unroll")
        self.__use_bias = hp.Boolean("use_bias")
        self.__kernel_initializer = hp.Choice(
            "kernel_initializer", ["glorot_uniform", "he_uniform", "he_normal"])
        self.__recurrent_initializer = hp.Choice(
            "recurrent_initializer", ["glorot_uniform", "he_uniform", "he_normal"])
        self.__bias_initializer = hp.Choice(
            "bias_initializer", ["zeros", "ones", "glorot_uniform"])

        self.__kernel_regularizer = get_regularizer(
            hp.Choice("kernel_regularizer", ["l1", "l2", "l1_l2"]))
        self.__recurrent_regularizer = get_regularizer(
            hp.Choice("recurrent_regularizer", ["l1", "l2", "l1_l2"]))
        self.__bias_regularizer = get_regularizer(
            hp.Choice("bias_regularizer", ["l1", "l2", "l1_l2"]))
        self.__activity_regularizer = get_regularizer(
            hp.Choice("activity_regularizer", ["l1", "l2", "l1_l2"]))

        self.__kernel_constraint = get_constraint(
            hp.Choice("kernel_constraint", ["max_norm", "non_neg", "unit_norm"]))
        self.__recurrent_constraint = get_constraint(
            hp.Choice("recurrent_constraint", ["max_norm", "non_neg", "unit_norm"]))
        self.__bias_constraint = get_constraint(
            hp.Choice("bias_constraint", ["max_norm", "non_neg", "unit_norm"]))

        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(
                units=self.__num_cats,
                input_shape=self.__input_shape,
                dropout=self.__dropout,
                recurrent_dropout=self.__recurrent_dropout,
                unroll=self.__unroll,
                use_bias=self.__use_bias,
                activation=self.__activation,
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
            tf.keras.layers.Dense(self.__num_cats, activation='softmax')
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def __update_best_args(self, acc: float, hparams: dict) -> None:
        """updates the best arguments found when a new model is better than the previous one.

        Args:
            new_accuracy (float): new accuracy found
            new_time (float): new execution time found
            hparams (dict): new hyperparameters found
        """
        if acc > self.__best_acc:
            self.__best_acc = acc
            print(f"New best accuracy: {acc:.4f}")
            if hparams:
                print(f"Best hyperparameters: {hparams}")

    def save_model(self) -> None:
        """"Saves the best model found.

        Raises:
            ValueError: if the model is None
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        os.makedirs("../models", exist_ok=True)
        self.__model.save(f"../models/best-rnn{self.__interval}.keras")
        self.__best_model_path = f"../models/best-rnn{self.__interval}.keras"

    def best_model(self) -> str:
        """Get the best model path.

        Returns:
            str: Path to the best model
        """
        return self.__best_model_path

    def confusion_matrix(self, filename: str, y_true: np.ndarray, y_pred: np.ndarray, tags: list[str]) -> str:
        """Generate a confusion matrix for the given model.

        Args:
            filename (str): Path to the model file
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            tags (list[str]): List of tags

        Returns:
            str: Path to the confusion matrix image
        """
        self.__model = tf.keras.models.load_model(filename)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        plot_filename = "./RNNlog/CM_" + \
            filename.split("/")[-1].replace(".keras", ".png")
        plot_confusion_matrix(y_true, y_pred, tags, plot_filename)
        plt.close()
        return plot_filename

    def stats(self) -> str:
        """Get the stats of the model.

        Returns:
            str: Stats of the model
        """
        return f"Best score: {self.__best_acc}"

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

    def get_log_dir(self) -> str:
        """Get the log directory.

        Returns:
            str: Log directory
        """
        return self.__tensorboard_log_dir


if __name__ == "__main__":
    X_train = np.random.randn(100, 10, 1)
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(100,)), num_classes=6)

    X_val = np.random.randn(20, 10, 1)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(20,)), num_classes=6)

    trainer = RNNTrainer("10")
    trainer.train_with_hparams(X_train, y_train, X_val, y_val,
                               epochs=5, batch_size=2, categories=[str(i) for i in range(6)])

    print(trainer.stats())

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", trainer.get_log_dir()])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
    while True:
        time.sleep(1)
