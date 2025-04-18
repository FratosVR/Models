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
                           epochs: int = 10, batch_size: int = 1, num_cats: int = 6, categories: list[str] = None) -> None:
        os.makedirs(self.__tensorboard_log_dir, exist_ok=True)

        completed_runs_path = os.path.join(
            self.__tensorboard_log_dir, f"completed_runs_{self.__interval}.json")
        if os.path.exists(completed_runs_path):
            with open(completed_runs_path, "r") as f:
                completed_runs = json.load(f)
        else:
            completed_runs = {}

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

            if run_name in completed_runs and completed_runs[run_name] == "done":
                print(f"{run_name} already completed, skipping.")
                session_num += 1
                continue

            print(f"Starting experiment {run_name} with {hparams}")

            self.__activation = hparam_values[0]
            self.__dropout = hparam_values[1]
            self.__recurrent_dropout = hparam_values[2]
            self.__recurrent_activation = hparam_values[3]
            self.__unroll = hparam_values[4]
            self.__use_bias = hparam_values[5]

            self.__model_generator(X[0].shape, num_cats)
            (acc, exec_time) = self.train(X, y, X_val, y_val, epochs,
                                          batch_size, session_log_dir, hparams)

            completed_runs[run_name] = {
                "status": "done", "acc": acc, "exec_time": exec_time}
            with open(completed_runs_path, "w") as f:
                json.dump(completed_runs, f, indent=2)

            session_num += 1
            del self.__model
            self.__model = None

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

        if self.__model is None:
            raise ValueError("Model is not initialized")

        self.__model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

        class CustomEarlyStopping(EarlyStopping):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.prev_losses = []

            def on_epoch_end(self, epoch, logs=None):
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

        early_stop = CustomEarlyStopping(monitor="val_loss", verbose=1)

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)

            self.__model.fit(
                X, y,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.__tensorboard_callbacks + [early_stop],
                verbose=1
            )

            _, accuracy = self.__model.evaluate(X_val, y_val)
            tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)

        time_start = time.time()
        self.__model.predict(X_val[0:1])
        time_end = time.time()

        if accuracy > self.__best_score:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
            self.save_model()
        elif accuracy == self.__best_score and (time_end - time_start) < self._exec_time:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
            self.save_model()

        return accuracy, time_end - time_start

    def __update_best_args(self, new_accuracy: float, new_time: float, hparams: dict) -> None:
        self.__best_args = hparams
        self._exec_time = new_time
        self.__best_score = new_accuracy

    def save_model(self) -> None:
        if self.__model is None:
            raise ValueError("Model is not initialized")
        os.makedirs("../models", exist_ok=True)
        self.__model.save(f"../models/best-lstm{self.__interval}.keras")
        self.__best_model_path = f"../models/best-lstm{self.__interval}.keras"

    def best_model(self):
        return self.__best_model_path

    def confusion_matrix(self, filename: str, y_true: np.ndarray, y_pred: np.ndarray, tags: list[str]) -> str:
        self.__model = tf.keras.models.load_model(filename)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        plot_filename = "./LSTMlog/CM_" + \
            filename.split("/")[-1].replace(".keras", ".png")
        plot_confusion_matrix(y_true, y_pred, tags, plot_filename)
        plt.close()
        return plot_filename

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
    trainer.train_with_hparams(X_train, y_train, X_val, y_val,
                               epochs=5, batch_size=2, categories=[str(i) for i in range(5)])

    print(trainer.stats())

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", trainer.get_log_dir()])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
