import tensorflow as tf
import numpy as np
import time
from itertools import product
import os
import json
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
from tensorboard import program
from tensorflow.keras.callbacks import EarlyStopping
from Utils import plot_confusion_matrix


class LSTMTrainer:
    """LSTM Model training manager with hyperparameter tuning using hparams."""

    def __init__(self, interval: float):
        """Constructor for LSTMTrainer.
        Initializes hyperparameters, HParams and TensorBoard log directory.

        Args:
            interval (float): Interval of which data is transmited to the model.
        """

        """Hparam for the dropout"""
        self.HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 1.0))
        """Hparam for the recurrent dropout"""
        self.HP_RECURRENT_DROPOUT = hp.HParam(
            'recurrent_dropout', hp.RealInterval(0.0, 1.0))
        """Hparam for the activation function"""
        self.HP_ACTIVATION = hp.HParam('activation', hp.Discrete(
            ["tanh", "linear", "relu", "sigmoid"]))
        """Hparam for the recurrent activation function"""
        self.HP_RECURRENT_ACTIVATION = hp.HParam(
            'recurrent_activation', hp.Discrete(["tanh", "linear", "relu", "sigmoid"]))
        """Hparam for the unroll"""
        self.HP_UNROLL = hp.HParam('unroll', hp.Discrete([True, False]))
        """Hparam for the use of bias"""
        self.HP_USE_BIAS = hp.HParam('use_bias', hp.Discrete([True, False]))
        """Metric for Hparam"""
        self.METRIC_ACCURACY = 'accuracy'
        """Interval of which data is transmited to the model"""
        self.__interval = interval
        """TensorBoard log directory"""
        self.__tensorboard_log_dir = f"./logs/hparams_LSTM{interval}"
        """TensorBoard callbacks for the model"""
        self.__tensorboard_callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.__tensorboard_log_dir, histogram_freq=1)
        ]

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
        """Generates a model with the given input and output shape.

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

        # Create the log directory if it doesn't exist
        os.makedirs(self.__tensorboard_log_dir, exist_ok=True)

        # read the already completed runs if there are any, if not create a new log
        completed_runs_path = os.path.join(
            self.__tensorboard_log_dir, f"completed_runs_{self.__interval}.json")
        if os.path.exists(completed_runs_path):
            with open(completed_runs_path, "r") as f:
                completed_runs = json.load(f)
        else:
            completed_runs = {}

        session_num = 0
        # make all combinations of hparams, create a set to avoid duplicates
        hparam_combinations = set(product(
            self.HP_ACTIVATION.domain.values,
            self._dropout_values,
            self._recurrent_dropout_values,
            self.HP_RECURRENT_ACTIVATION.domain.values,
            self.HP_UNROLL.domain.values,
            self.HP_USE_BIAS.domain.values
        ))

        # train with every combination
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
            # create session log directory
            session_log_dir = os.path.join(
                self.__tensorboard_log_dir, run_name)

            # if run already completed, skip
            if run_name in completed_runs and completed_runs[run_name] == "done":
                print(f"{run_name} already completed, skipping.")
                session_num += 1
                continue

            print(f"Starting train {run_name} with {hparams}")

            # set hyperparameters
            self.__activation = hparam_values[0]
            self.__dropout = hparam_values[1]
            self.__recurrent_dropout = hparam_values[2]
            self.__recurrent_activation = hparam_values[3]
            self.__unroll = hparam_values[4]
            self.__use_bias = hparam_values[5]

            # generate model
            self.__model_generator(X[0].shape, num_cats)

            # train model
            (acc, exec_time) = self.train(X, y, X_val, y_val, epochs,
                                          batch_size, session_log_dir, hparams)
            # log results
            completed_runs[run_name] = {
                "status": "done", "acc": acc, "exec_time": exec_time}

            # write log to final log
            with open(completed_runs_path, "w") as f:
                json.dump(completed_runs, f, indent=2)

            session_num += 1
            # delete model to free memory
            del self.__model
            self.__model = None

        # write the best model to a file
        os.makedirs("./LSTMlog", exist_ok=True)
        f = open(os.path.join(
            "./LSTMlog", f"best_model_{self.__interval}.txt"), "w")
        f.write(f"Best model: {self.__best_model_path}\n")
        f.write(f"Best score: {self.__best_score}\n")
        f.write(f"Best args: {self.__best_args}\n")
        f.write(f"Execution time: {self._exec_time}\n")
        f.close()

        self.confusion_matrix(
            self.__best_model_path,
            y_true=np.concatenate((y, y_val)),
            y_pred=np.concatenate((self.predict(X), self.predict(X_val))),
            tags=categories
        )

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 1, log_dir: str = None, hparams: dict = None) -> list[float]:
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

        Raises:
            ValueError: If the model is not initialized
        """

        # if the model is none, exception
        if self.__model is None:
            raise ValueError("Model is not initialized")

        self.__model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

        class CustomEarlyStopping(EarlyStopping):
            """Custom early stopping class to stop training when the accuracy does not improve."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.prev_losses = []

            def on_epoch_end(self, epoch, logs=None):
                # Evaluate the accuracy of the last two epochs. if there is no improvement, stop training
                current_loss = logs.get("accuracy")
                self.prev_losses.append(current_loss)
                if len(self.prev_losses) >= 2:
                    loss1, loss2 = self.prev_losses[-2], self.prev_losses[-1]
                    if loss1 is not None and loss2 is not None:
                        improvement = (loss1 - loss2) / (loss1 + 1e-8)
                        if improvement == 0:
                            print(
                                f"Early stopping at epoch {epoch+1}: Improvement {improvement*100:.2f}%")
                            self.model.stop_training = True

        early_stop = CustomEarlyStopping(monitor="accuracy", verbose=1)

        # log training with tensorboard
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)

            # train the model
            self.__model.fit(
                X, y,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.__tensorboard_callbacks + [early_stop],
                verbose=1
            )

            # log the accuracy to tensorboard
            _, accuracy = self.__model.evaluate(X_val, y_val)
            tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)

        # get the prediction time for one example
        time_start = time.time()
        self.__model.predict(X_val[0:1])
        time_end = time.time()

        # Evaluate possible best model
        if accuracy > self.__best_score:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
            self.save_model()
        elif accuracy == self.__best_score and (time_end - time_start) < self._exec_time:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
            self.save_model()

        return accuracy, time_end - time_start

    def __update_best_args(self, new_accuracy: float, new_time: float, hparams: dict) -> None:
        """Update the best arguments with the new accuracy and time.

        Args:
            new_accuracy (float): New accuracy
            new_time (float): New time
            hparams (dict): Hyperparameters
        """
        self.__best_args = hparams
        self._exec_time = new_time
        self.__best_score = new_accuracy

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

    def stats(self) -> str:
        """Get the stats of the model.

        Returns:
            str: Stats of the model
        """
        return f"Best score: {self.__best_score}, Best args: {self.__best_args}, Execution time: {self._exec_time}"

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
        np.random.randint(0, 5, size=(100,)), num_classes=5)

    X_val = np.random.randn(20, 10, 1)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 5, size=(20,)), num_classes=5)

    trainer = LSTMTrainer(10)
    trainer.train_with_hparams(X_train, y_train, X_val, y_val,
                               epochs=5, batch_size=2, categories=[str(i) for i in range(5)])

    print(trainer.stats())

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", trainer.get_log_dir()])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
