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
        self.__exec_time = float('inf')
        """Best model path."""
        self.__best_model_path = f"../model/best-rnn{self.__interval}.h5"

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

        # Generate all possible combinations of hyperparameters
        hparams_combinations = set(product(
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
        os.makedirs("./RNNLog", exist_ok=True)
        f = open(os.path.join(
            "./RNNlog", f"best_model_{self.__interval}.txt"), "w")
        f.write(f"Best model: {self.__best_model_path}\n")
        f.write(f"Best score: {self.__best_score}\n")
        f.write(f"Best args: {self.__best_args}\n")
        f.write(f"Execution time: {self.__exec_time}\n")
        f.close()

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
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

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

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)
            history = self.__model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=self.__tensorboard_callbacks + [early_stop],
                verbose=1
            )

            # log the accuracy to tensorboard
            _, accuracy = self.__model.evaluate(X_val, y_val)
            tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)

        # get the prediction time for one example
        time_start: float = time.time()
        self.__model.predict(X_val[0])
        time_end: float = time.time()

        # Evaluate possible best model
        if accuracy > self.__best_score:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
            self.save_model()
        elif accuracy == self.__best_score and (time_end - time_start) < self.__exec_time:
            self.__update_best_args(accuracy, time_end - time_start, hparams)
            self.save_model()

    def __update_best_args(self, new_accuracy: float, new_time: float, hparams: dict) -> None:
        """updates the best arguments found when a new model is better than the previous one.

        Args:
            new_accuracy (float): new accuracy found
            new_time (float): new execution time found
            hparams (dict): new hyperparameters found
        """
        self.__best_score = new_accuracy
        self.__exec_time = new_time
        self.__best_args = hparams

    def save_model(self) -> None:
        """"Saves the best model found.

        Raises:
            ValueError: if the model is None
        """
        if self.__model is None:
            raise ValueError("Model is not initialized")
        self.__model.save(f"../model/best-rnn{self.__interval}.h5")

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
        return f"Best score: {self.__best_score}, Best args: {self.__best_args}, Execution time: {self.__exec_time}"

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
    y_train = np.random.randn(0, 5, size=(100, 5))

    X_val = np.random.rand(20, 10, 1)
    y_val = np.random.rand(0, 5, size=(20, 5))

    trainer = RNNTrainer(10)
    trainer.train_with_hparams(X_train, y_train, X_val, y_val)

    print(f"Best hyperparameters found: {trainer.__best_args}")
