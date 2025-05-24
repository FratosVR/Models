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
    """RNN Model training manager with hyperparameter tuning using hparams and keras tuner.
    
    Supports SimpleRNN architecture with comprehensive hyperparameter optimization.
    """

    def __init__(self, interval: str, tensorboard_log_dir: str = "./logs/hparams_RNN") -> None:
        """Initialize the RNNTrainer instance.

        :param interval: Data sampling interval identifier
        :type interval: str
        :param tensorboard_log_dir: Directory for TensorBoard logs, defaults to "./logs/hparams_RNN"
        :type tensorboard_log_dir: str, optional
        """
        self.__interval: str = interval  #: Data interval identifier
        self.__model: tf.keras.Model = None  #: Keras model instance
        #: Path to best saved model
        self.__best_model_path: str = f"best_rnn_{interval}.keras"
        #: Directory for TensorBoard logs
        self.__tensorboard_log_dir: str = tensorboard_log_dir
        #: List of TensorBoard callback instances
        self.__tensorboard_callbacks: list[TensorBoard] = [
            TensorBoard(log_dir=os.path.join(tensorboard_log_dir, interval))
        ]
        self.__best_acc: float = 0.0  #: Best achieved accuracy
        self.__tuner: Hyperband = None  #: Keras tuner instance
        self.__input_shape: tuple = None  #: Input shape of the model
        self.__num_cats: int = None  #: Number of output categories
        # RNN hyperparameters
        self.__dropout: float = None  #: Dropout rate
        self.__recurrent_dropout: float = None  #: Recurrent dropout rate
        self.__activation: str = None  #: Activation function
        self.__unroll: bool = None  #: Whether to unroll RNN
        self.__use_bias: bool = None  #: Whether to use bias
        self.__kernel_initializer: str = None  #: Kernel initializer
        self.__recurrent_initializer: str = None  #: Recurrent initializer
        self.__bias_initializer: str = None  #: Bias initializer
        self.__kernel_regularizer: str = None  #: Kernel regularizer
        self.__recurrent_regularizer: str = None  #: Recurrent regularizer
        self.__bias_regularizer: str = None  #: Bias regularizer
        self.__activity_regularizer: str = None  #: Activity regularizer
        self.__kernel_constraint: str = None  #: Kernel constraint
        self.__recurrent_constraint: str = None  #: Recurrent constraint
        self.__bias_constraint: str = None  #: Bias constraint

    def __model_generator(self, input_shape: tuple[int, int], output_shape: int) -> None:
        """LEGACY - Generate RNN model with given specifications.

        :param input_shape: Input shape (timesteps, features)
        :type input_shape: tuple[int, int]
        :param output_shape: Number of output classes
        :type output_shape: int
        """
        self.__model = tf.keras.Sequential([
            tf.keras.layers.RNN(
                units=output_shape,
                input_shape=input_shape,
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

    def train_with_hparams(self, X: np.ndarray, y: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None,
                          X_test: np.ndarray = None, y_test: np.ndarray = None,
                          epochs: int = 10, batch_size: int = 1,
                          num_cats: int = 6, categories: list[str] = None) -> None:
        """Train model with hyperparameter tuning using Keras Tuner.

        :param X: Training input data
        :type X: np.ndarray
        :param y: Training target data (one-hot encoded)
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
            project_name=f"tune_rnn_{self.__interval}",
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
        :param y: Training target data (one-hot encoded)
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
            tf.keras.layers.SimpleRNN(units=self.__num_cats, input_shape=input_shape),
            tf.keras.layers.Dense(self.__num_cats, activation="softmax")
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
        self.__activation = hp.Choice("activation", ["tanh", "linear", "relu", "sigmoid"])
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
        self.__model.save(f"../models/best-rnn{self.__interval}.keras")
        self.__best_model_path = f"../models/best-rnn{self.__interval}.keras"

    def best_model(self) -> str:
        """Get path to best saved model.

        :return: Path to model file
        :rtype: str
        """
        return self.__best_model_path

    def confusion_matrix(self, filename: str, y_true: np.ndarray,
                         y_pred: np.ndarray, tags: list[str]) -> str:
        """Generate and save confusion matrix plot.

        :param filename: Model filename base
        :type filename: str
        :param y_true: True labels (one-hot encoded)
        :type y_true: np.ndarray
        :param y_pred: Predicted labels (one-hot encoded)
        :type y_pred: np.ndarray
        :param tags: Category names for labeling
        :type tags: list[str]
        :return: Path to saved confusion matrix image
        :rtype: str
        """
        self.__model = tf.keras.models.load_model(filename)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        plot_filename = os.path.join("./RNNlog", f"CM_{filename.split('/')[-1].replace('.keras', '.png')}")
        plot_confusion_matrix(
            y_true, y_pred, tags, plot_filename,
            title=f"Confusion Matrix (RNN, interval {self.__interval}s)"
        )
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

#     trainer = RNNTrainer("10")
#     trainer.train_with_hparams(X_train, y_train, X_val, y_val,
#                                epochs=5, batch_size=2, categories=[str(i) for i in range(6)])

#     print(trainer.stats())

#     tb = program.TensorBoard()
#     tb.configure(argv=[None, "--logdir", trainer.get_log_dir()])
#     url = tb.launch()
#     print(f"TensorBoard started at {url}")
#     while True:
#         time.sleep(1)
