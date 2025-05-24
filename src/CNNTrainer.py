import tensorflow as tf
import numpy as np
import time
import os
import json
from itertools import product
from tensorboard.plugins.hparams import api as hp
from tensorboard import program
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D, MaxPooling1D,
    GlobalAveragePooling1D,
    Dense, Dropout
)
from Utils import plot_confusion_matrix
from keras_tuner import HyperParameters, Hyperband
import matplotlib.pyplot as plt


class CNNTrainer:
    """CNN Model training manager with hyperparameter tuning using hparams and keras tuner.
    
    Handles variable-length input sequences for animation classification.
    """

    def __init__(self, interval: float, tensorboard_log_dir: str = "./logs/hparams_CNN") -> None:
        """Initialize the CNNTrainer instance.

        :param interval: Data sampling interval in seconds
        :type interval: float
        :param tensorboard_log_dir: Directory for TensorBoard logs, defaults to "./logs/hparams_CNN"
        :type tensorboard_log_dir: str, optional
        """
        self.__interval: float = interval  #: Data sampling interval
        self.__model: Model = None  #: Keras model instance
        #: Path to best saved model
        self.__best_model_path: str = f"best_cnn_{interval}.keras"
        #: Directory for TensorBoard logs
        self.__tensorboard_log_dir: str = tensorboard_log_dir
        #: List of TensorBoard callback instances
        self.__tensorboard_callbacks: list[TensorBoard] = [
            TensorBoard(log_dir=os.path.join(tensorboard_log_dir, str(interval)))
        ]
        self.__best_acc: float = 0.0  #: Best achieved accuracy
        self.__tuner: Hyperband = None  #: Keras tuner instance
        self.__input_shape: tuple = None  #: Input shape of the model
        self.__num_cats: int = None  #: Number of output categories
        #: Hyperparameters
        self.__conv_filters: int = None
        self.__kernel_size: int = None
        self.__activation: str = None
        self.__pool_size: int = None
        self.__dense_units: int = None
        self.__dropout_rate: float = None

    def __model_generator(self, n_features: int, n_classes: int) -> None:
        """LEGACY - Generate a CNN model with given specifications.

        :param n_features: Number of input features per timestep
        :type n_features: int
        :param n_classes: Number of output classes
        :type n_classes: int
        """
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
                           X_test: np.ndarray = None, y_test: np.ndarray = None,
                           epochs: int = 10, batch_size: int = 1,
                           num_cats: int = 6, categories: list[str] = None) -> None:
        """Train model with hyperparameter tuning using Keras Tuner.

        :param X: Training input data
        :type X: np.ndarray
        :param y: Training target data
        :type y: np.ndarray
        :param X_val: Validation input data, defaults to None
        :type X_val: np.ndarray, optional
        :param y_val: Validation target data, defaults to None
        :type y_val: np.ndarray, optional
        :param X_test: Test input data, defaults to None
        :type X_test: np.ndarray, optional
        :param y_test: Test target data, defaults to None
        :type y_test: np.ndarray, optional
        :param epochs: Number of training epochs, defaults to 10
        :type epochs: int, optional
        :param batch_size: Batch size, defaults to 1
        :type batch_size: int, optional
        :param num_cats: Number of output categories, defaults to 6
        :type num_cats: int, optional
        :param categories: List of category names, defaults to None
        :type categories: list[str], optional
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
            project_name=f"tune_cnn_{self.__interval}",
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
        self.__update_best_args(best_model.evaluate(X_test, y_test)[1], best_hp.values)
        self.save_model()

        self.__cm_file_path = self.confusion_matrix(
            self.__best_model_path,
            y_true=np.concatenate((y, y_val)),
            y_pred=np.concatenate((self.predict(X), self.predict(X_val), self.predict(X_test))),
            tags=categories
        )

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 1,
              log_dir: str = None, hparams: dict = None) -> None:
        """Train model with fixed hyperparameters.

        :param X: Training input data
        :type X: np.ndarray
        :param y: Training target data
        :type y: np.ndarray
        :param X_val: Validation input data, defaults to None
        :type X_val: np.ndarray, optional
        :param y_val: Validation target data, defaults to None
        :type y_val: np.ndarray, optional
        :param epochs: Number of training epochs, defaults to 10
        :type epochs: int, optional
        :param batch_size: Batch size, defaults to 1
        :type batch_size: int, optional
        :param log_dir: Directory for training logs, defaults to None
        :type log_dir: str, optional
        :param hparams: Hyperparameters dictionary, defaults to None
        :type hparams: dict, optional
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

        y_pred = [self.__model.predict(x) for x in np.concatenate((X, X_val), axis=1)]

        self.__cm_file_path = self.confusion_matrix(
            self.__best_model_path,
            y_true=np.concatenate((y, y_val)),
            y_pred=y_pred,
            tags=categories
        )

    def __build_model(self, hp: HyperParameters) -> tf.keras.Model:
        """Build model architecture with tunable hyperparameters.

        :param hp: Hyperparameters configuration
        :type hp: HyperParameters
        :return: Compiled Keras model
        :rtype: tf.keras.Model
        """
        self.__conv_filters = hp.Int("conv_filters", 16, 64, step=16)
        self.__kernel_size = hp.Int("kernel_size", 2, 5, step=1)
        self.__activation = hp.Choice("activation", ["relu", "tanh"])
        self.__pool_size = hp.Int("pool_size", 2, 4, step=1)
        self.__dense_units = hp.Int("dense_units", 32, 128, step=16)
        self.__dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)

        model = tf.keras.Sequential()
        model.add(Input(shape=self.__input_shape))
        model.add(Conv1D(filters=self.__conv_filters,
                         kernel_size=self.__kernel_size,
                         activation=self.__activation,
                         padding='same'))
        model.add(MaxPooling1D(pool_size=self.__pool_size, padding='same'))

        model.add(Conv1D(filters=self.__conv_filters * 2,
                         kernel_size=self.__kernel_size,
                         activation=self.__activation,
                         padding='same'))
        model.add(MaxPooling1D(pool_size=self.__pool_size, padding='same'))

        model.add(Conv1D(filters=self.__conv_filters * 2,
                         kernel_size=self.__kernel_size,
                         activation=self.__activation,
                         padding='same'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(self.__dense_units, activation=self.__activation))
        model.add(Dropout(self.__dropout_rate))
        model.add(Dense(self.__num_cats, activation='softmax'))

        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def __update_best_args(self, acc: float, hparams: dict = None) -> None:
        """Update tracking of best model metrics.

        :param acc: New accuracy value
        :type acc: float
        :param hparams: Hyperparameters dictionary, defaults to None
        :type hparams: dict, optional
        """
        if acc > self.__best_acc:
            self.__best_acc = acc
            print(f"New best accuracy: {acc:.4f}")
            if hparams:
                print(f"Best hyperparameters: {hparams}")

    def save_model(self) -> None:
        """Save current model to disk.

        :raises ValueError: If model is not initialized
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        os.makedirs("../models", exist_ok=True)
        self.__model.save(f"../models/best-cnn{self.__interval}.keras")
        self.__best_model_path = f"../models/best-cnn{self.__interval}.keras"

    def best_model(self) -> str:
        """Get path to best saved model.

        :return: Path to model file
        :rtype: str
        """
        return self.__best_model_path

    def confusion_matrix(self, filename: str, y_true: np.ndarray, 
                         y_pred: np.ndarray, tags: list[str]) -> str:
        """Generate and save confusion matrix.

        :param filename: Model filename
        :type filename: str
        :param y_true: True labels
        :type y_true: np.ndarray
        :param y_pred: Predicted labels
        :type y_pred: np.ndarray
        :param tags: Category names
        :type tags: list[str]
        :return: Path to saved confusion matrix image
        :rtype: str
        """
        self.__model = tf.keras.models.load_model(filename)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        plot_filename = "./CNNlog/CM_" + filename.split("/")[-1].replace(".keras", ".png")
        plot_confusion_matrix(y_true, y_pred, tags, plot_filename,
                            title=f"Confusion Matrix (CNN, interval {self.__interval}s)")
        plt.close()
        return plot_filename

    def get_best_acc(self) -> float:
        """Get best achieved accuracy.

        :return: Accuracy value
        :rtype: float
        """
        return self.__best_acc

    def get_confusion_matrix(self) -> str:
        """Get path to confusion matrix image.

        :return: Path to image file
        :rtype: str
        """
        return self.__cm_file_path

    def stats(self) -> str:
        """Get training statistics summary.

        :return: Formatted statistics string
        :rtype: str
        """
        return f"Best score: {self.__best_acc}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input data.

        :param X: Input data
        :type X: np.ndarray
        :return: Model predictions
        :rtype: np.ndarray
        :raises ValueError: If model is not initialized
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        return self.__model.predict(X)

    def get_log_dir(self) -> str:
        """Get TensorBoard log directory.

        :return: Path to log directory
        :rtype: str
        """
        return self.__tensorboard_log_dir

# Example usage:
# Uncomment the following lines to run the example
# if __name__ == "__main__":
#     X_train = np.random.randn(100, 10, 1)
#     y_train = tf.keras.utils.to_categorical(
#         np.random.randint(0, 6, size=(100,)), num_classes=6)

#     X_val = np.random.randn(20, 10, 1)
#     y_val = tf.keras.utils.to_categorical(
#         np.random.randint(0, 6, size=(20,)), num_classes=6)

#     trainer = CNNTrainer('10')
#     trainer.train_with_hparams(X_train, y_train, X_val, y_val,
#                                epochs=5, batch_size=2, categories=[str(i) for i in range(6)])

#     print(trainer.stats())

#     tb = program.TensorBoard()
#     tb.configure(argv=[None, "--logdir", trainer.get_log_dir()])
#     url = tb.launch()
#     print(f"TensorBoard started at {url}")
#     while True:
#         time.sleep(1)
