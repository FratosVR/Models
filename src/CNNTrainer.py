import tensorflow as tf
import numpy as np
import time
import os
import json
from itertools import product
from tensorboard.plugins.hparams import api as hp
from tensorboard import program
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D, MaxPooling1D,
    GlobalAveragePooling1D,
    Dense, Dropout
)
from Utils import plot_confusion_matrix


class CNNTrainer:
    """CNN Model training manager with hyperparameter tuning using hparams, accepts variable-length input."""

    def __init__(self, interval: float):
        # Hyperparameter definitions
        self.HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3, 5]))
        self.HP_POOL_SIZE = hp.HParam('pool_size', hp.Discrete([2, 3]))
        self.HP_CONV_FILTERS = hp.HParam(
            'conv_filters', hp.Discrete([32, 64, 128]))
        self.HP_DENSE_UNITS = hp.HParam(
            'dense_units', hp.Discrete([64, 128, 256]))
        self.HP_DROPOUT_RATE = hp.HParam(
            'dropout_rate', hp.Discrete([0.0, 0.2, 0.5]))
        self.HP_ACTIVATION = hp.HParam(
            'activation', hp.Discrete(['relu', 'elu', 'tanh']))
        self.HP_OPTIMIZER = hp.HParam(
            'optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
        self.HP_LEARNING_RATE = hp.HParam(
            'learning_rate', hp.Discrete([0.0001, 0.001, 0.01]))

        self.METRIC_ACCURACY = 'accuracy'
        self.__interval = interval
        self.__tensorboard_log_dir = f"./logs/hparams_CNN{interval}"
        self.__tensorboard_callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.__tensorboard_log_dir, histogram_freq=1)
        ]

        # Default hyperparameter values
        self.__conv_filters = 32
        self.__kernel_size = 3
        self.__pool_size = 2
        self.__dense_units = 64
        self.__dropout_rate = 0.0
        self.__activation = 'relu'
        self.__optimizer = 'adam'
        self.__learning_rate = 0.001
        self.__model = None
        self.__best_score = 0.0
        self.__exec_time = float('inf')
        self.__best_args = {}
        self.__best_model_path = None

    def __model_generator(self, n_features: int, n_classes: int) -> None:
        """Builds a Conv1D+GlobalAveragePooling1D model that accepts variable-length sequences."""
        inp = Input(shape=(None, n_features), name="animation_frames")
        x = Conv1D(filters=self.__conv_filters,
                   kernel_size=self.__kernel_size,
                   activation=self.__activation,
                   padding='same')(inp)
        x = MaxPooling1D(pool_size=self.__pool_size, padding='same')(x)

        x = Conv1D(filters=self.__conv_filters * 2,
                   kernel_size=self.__kernel_size,
                   activation=self.__activation,
                   padding='same')(x)
        x = MaxPooling1D(pool_size=self.__pool_size, padding='same')(x)

        x = Conv1D(filters=self.__conv_filters * 2,
                   kernel_size=self.__kernel_size,
                   activation=self.__activation,
                   padding='same')(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(self.__dense_units, activation=self.__activation)(x)
        x = Dropout(self.__dropout_rate)(x)
        out = Dense(n_classes, activation='softmax')(x)

        self.__model = Model(inputs=inp, outputs=out)

    def train_with_hparams(self, X: np.ndarray, y: np.ndarray,
                           X_val: np.ndarray = None, y_val: np.ndarray = None,
                           epochs: int = 10, batch_size: int = 1,
                           num_cats: int = 6, categories: list[str] = None) -> None:
        os.makedirs(self.__tensorboard_log_dir, exist_ok=True)
        completed_runs_path = os.path.join(
            self.__tensorboard_log_dir, f"completed_runs_{self.__interval}.json")

        completed_runs = {}
        if os.path.exists(completed_runs_path):
            with open(completed_runs_path, "r") as f:
                completed_runs = json.load(f)

        session_num = 0
        hparam_combinations = list(product(
            self.HP_CONV_FILTERS.domain.values,
            self.HP_KERNEL_SIZE.domain.values,
            self.HP_POOL_SIZE.domain.values,
            self.HP_DENSE_UNITS.domain.values,
            self.HP_DROPOUT_RATE.domain.values,
            self.HP_ACTIVATION.domain.values,
            self.HP_OPTIMIZER.domain.values,
            self.HP_LEARNING_RATE.domain.values
        ))

        n_features = X.shape[-1]
        for hparam_values in hparam_combinations:
            hparams = {
                self.HP_CONV_FILTERS: hparam_values[0],
                self.HP_KERNEL_SIZE: hparam_values[1],
                self.HP_POOL_SIZE: hparam_values[2],
                self.HP_DENSE_UNITS: hparam_values[3],
                self.HP_DROPOUT_RATE: hparam_values[4],
                self.HP_ACTIVATION: hparam_values[5],
                self.HP_OPTIMIZER: hparam_values[6],
                self.HP_LEARNING_RATE: hparam_values[7]
            }
            run_name = f"run-{session_num}"
            session_log_dir = os.path.join(
                self.__tensorboard_log_dir, run_name)

            if run_name in completed_runs and completed_runs[run_name].get('status') == 'done':
                print(f"Skipping completed {run_name}")
                session_num += 1
                continue

            print(f"Starting {run_name} with hparams: {hparams}")
            (self.__conv_filters, self.__kernel_size,
             self.__pool_size, self.__dense_units,
             self.__dropout_rate, self.__activation,
             self.__optimizer, self.__learning_rate) = hparam_values

            self.__model_generator(n_features, num_cats)
            acc, exec_time = self.train(
                X, y, X_val, y_val, epochs, batch_size,
                session_log_dir, hparams)

            completed_runs[run_name] = {
                "status": "done", "acc": acc, "exec_time": exec_time}
            with open(completed_runs_path, "w") as f:
                json.dump(completed_runs, f, indent=2)

            session_num += 1
            del self.__model
            self.__model = None

        self._save_best_model_report(categories, X, X_val, y, y_val)

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 1,
              log_dir: str = None, hparams: dict = None) -> tuple:
        if self.__model is None:
            raise ValueError("Model not initialized")

        if self.__optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)
        elif self.__optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=self.__learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(
                learning_rate=self.__learning_rate)

        self.__model.compile(optimizer=opt,
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        early_stop = EarlyStopping(
            monitor='val_accuracy', patience=2, restore_best_weights=True)

        start_time = time.time()
        history = self.__model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.__tensorboard_callbacks + [early_stop],
            verbose=0
        )

        _, accuracy = self.__model.evaluate(X_val, y_val, verbose=0)
        exec_time = time.time() - start_time

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)
            tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)

        if accuracy > self.__best_score or (
                accuracy == self.__best_score and exec_time < self.__exec_time):
            self.__best_score = accuracy
            self.__exec_time = exec_time
            self.__best_args = hparams
            self.save_model()

        return accuracy, exec_time

    def save_model(self) -> None:
        os.makedirs("../models", exist_ok=True)
        model_path = f"../models/best-cnn{self.__interval}.keras"
        self.__model.save(model_path)
        self.__best_model_path = model_path

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__model.predict(X)

    def stats(self) -> str:
        return f"Best Accuracy: {self.__best_score:.4f}, Time: {self.__exec_time:.2f}s, Params: {self.__best_args}"

    def get_log_dir(self) -> str:
        return self.__tensorboard_log_dir


if __name__ == "__main__":
    # Example usage with variable-length sequences
    X_train = np.random.randn(100, np.random.randint(5, 10), 56)
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(100,)), num_classes=6)

    X_val = np.random.randn(20, np.random.randint(5, 10), 56)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(20,)), num_classes=6)

    trainer = CNNTrainer(interval=10)
    trainer.train_with_hparams(
        X_train, y_train, X_val, y_val,
        epochs=5, batch_size=2, num_cats=6,
        categories=[str(i) for i in range(6)]
    )

    print(trainer.stats())

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", trainer.get_log_dir()])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
