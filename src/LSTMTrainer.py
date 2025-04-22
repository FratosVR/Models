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
from keras_tuner import HyperParameters, Hyperband


class LSTMTrainer:
    """LSTM Model training manager with hyperparameter tuning using hparams and keras tune."""

    def __init__(self, interval: str, tensorboard_log_dir: str = "./logs/hparams_LTSM") -> None:
        """Constructor for LSTMTrainer.
        Initializes hyperparameters, HParams and TensorBoard log directory.

        Args:
            interval (float): Interval of which data is transmited to the model.
            tensorboard_log_dir (str, optional): Directory for TensorBoard logs.
        """
        self.__interval = interval
        self.__model = None
        self.__best_model_path = f"best_ltsm_{interval}.keras"
        self.__tensorboard_log_dir = tensorboard_log_dir
        self.__tensorboard_callbacks = [TensorBoard(
            log_dir=os.path.join(tensorboard_log_dir, interval))]
        self.__best_acc = 0.0
        self.__tuner = None

    def __model_generator(self, input_shape: tuple[int, int], output_shape: int) -> None:
        """LEGACY 

        Generates a model with the given input and output shape.

        Args:
            input_shape (tuple[int, int]): input shape
            output_shape (int): output shape
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
            max_epochs=epochs * 3,
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
              epochs: int = 10, batch_size: int = 1, num_cats: int = 6, categories: list[str] = None) -> None:
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

        Returns:
            list[float]: Accuracy and execution time
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

        y_pred = [self.__model.predict(x)
                  for x in np.concatenate((X, X_val), axis=1)]

        self.confusion_matrix(
            self.__best_model_path,
            y_true=np.concatenate((y, y_val)),
            y_pred=y_pred,
            tags=categories
        )

    def __build_model(self, hp: HyperParameters) -> tf.keras.Model:
        """Build the model with the given hyperparameters.
        Args:
            hp (HyperParameters): Hyperparameters (automatically created by keras tuner)
        Returns:
            tf.keras.Model: Model
        """
        self.__dropout = hp.Float("dropout", 0.0, 1.0)
        self.__recurrent_dropout = hp.Float("recurrent_dropout", 0.0, 1.0)
        self.__activation = hp.Choice(
            "activation", ["tanh", "linear", "relu", "sigmoid"])
        self.__recurrent_activation = hp.Choice(
            "recurrent_activation", ["tanh", "linear", "relu", "sigmoid"])
        self.__unroll = hp.Boolean("unroll")
        self.__use_bias = hp.Boolean("use_bias")

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=self.__num_cats,
                input_shape=self.__input_shape,
                dropout=self.__dropout,
                recurrent_dropout=self.__recurrent_dropout,
                activation=self.__activation,
                recurrent_activation=self.__recurrent_activation,
                unroll=self.__unroll,
                use_bias=self.__use_bias
            ),
            tf.keras.layers.Dense(self.__num_cats, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def __update_best_args(self, acc: float, hparams: dict = None) -> None:
        """Update the best arguments with the new accuracy and time.

        Args:
            new_accuracy (float): New accuracy
            new_time (float): New time
            hparams (dict): Hyperparameters
        """
        if acc > self.__best_acc:
            self.__best_acc = acc
            print(f"New best accuracy: {acc:.4f}")
            if hparams:
                print(f"Best hyperparameters: {hparams}")

    def save_model(self) -> None:
        """Save the model to a file.

        Raises:
            ValueError: If the model is not initialized
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        os.makedirs("../models", exist_ok=True)
        self.__model.save(f"../models/best-lstm{self.__interval}.keras")
        self.__best_model_path = f"../models/best-lstm{self.__interval}.keras"

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
        plot_filename = "./LSTMlog/CM_" + \
            filename.split("/")[-1].replace(".keras", ".png")
        plot_confusion_matrix(y_true, y_pred, tags, plot_filename)
        plt.close()
        return plot_filename

    def get_best_acc(self) -> float:
        """Get the best accuracy.

        Returns:
            float: Best accuracy
        """
        return self.__best_acc

    def get_confusion_matrix(self) -> str:
        """Get the confusion matrix.

        Returns:
            str: Path to the confusion matrix image
        """
        return self.__best_model_path.replace(".keras", ".png")

    def stats(self) -> str:
        """Get the stats of the model.

        Returns:
            str: Stats of the model
        """
        return f"Best score: {self.__best_acc}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the output for the given input.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Predicted output

        Raises:
            ValueError: If the model is not initialized
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

    trainer = LSTMTrainer('10')
    trainer.train_with_hparams(X_train, y_train, X_val, y_val,
                               epochs=5, batch_size=2, categories=[str(i) for i in range(6)])

    print(trainer.stats())

    trainer.predict(X_val[0:1])

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", trainer.get_log_dir()])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
    while True:
        time.sleep(1)
