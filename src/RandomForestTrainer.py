import os
import numpy as np
import pandas as pd
import time
import ydf
import Utils
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
import pickle
import json

ydf.verbose(2)


class RandomForestTrainer:
    """Random Forest trainer using Yggdrasil Decision Forests (ydf) with manual hyperparameter tuning and TensorBoard logging."""

    def __init__(self, interval: str, log_dir: str = "./RFlog") -> None:
        self.__interval = interval
        self.__model = None
        self.__best_model_path = f"./models/best_rf_{interval}"
        self.__cm_file_path = None
        self.__log_dir = log_dir
        self.__best_accuracy = 0.0
        self.__tensorboard_callbacks = [TensorBoard(
            log_dir=os.path.join(log_dir, f"rf_{interval}"))]

    def train_with_hparams(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, X_test: np.ndarray = None, y_test: np.ndarray = None,
                           epochs: int = 10, batch_size: int = 1, num_cats: int = 6, categories: list[str] = None) -> None:
        """Manually search for the best hyperparameters for the Random Forest model."""

        X_test = np.concatenate((X_test, X_val))
        y_test = np.concatenate((y_test, y_val))

        best_acc = 0.0
        best_model = None
        best_params = {}

        # Prepare data for YDF (convert to pandas DataFrame)
        X_reshaped = X.reshape(X.shape[0], -1)
        df_train = pd.DataFrame(
            X_reshaped, columns=[f"feature_{i}" for i in range(X_reshaped.shape[1])])

        X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
        df_val = pd.DataFrame(
            X_val_reshaped, columns=[f"feature_{i}" for i in range(X_val_reshaped.shape[1])])

        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        df_test = pd.DataFrame(
            X_test_reshaped, columns=[f"feature_{i}" for i in range(X_test_reshaped.shape[1])])

        # Ensure 'label' column is added and is of string type
        df_train['label'] = np.argmax(y, axis=1)
        df_val['label'] = np.argmax(y_val, axis=1)
        df_test['label'] = np.argmax(y_test, axis=1)

        tuner = ydf.RandomSearchTuner(
            num_trials=50, automatic_search_space=False)

        tuner.choice('num_trees', [50, 100, 150, 200])
        tuner.choice('max_depth', [3, 5, 10, 15])
        tuner.choice('min_examples', [2, 5, 8, 10])
        # tuner.choice('categorical_algorithm', ['CART', 'RANDOM'])
        # tuner.choice('honest', [True, False])
        # tuner.choice('split_axis', ['AXIS_ALIGNED',
        #              'SPARSE_OBLIQUE'])

        # Create and train the Random Forest model
        learner = ydf.RandomForestLearner(
            label="label",
            tuner=tuner
        )

        model = learner.train(df_train)
        eval_result = model.evaluate(df_test)
        acc = eval_result.accuracy

        test_acc = model.evaluate(df_test).accuracy
        test_loss = model.evaluate(df_test).loss

        train_acc = model.evaluate(df_train).accuracy
        train_loss = model.evaluate(df_train).loss

        val_acc = model.evaluate(df_val).accuracy
        val_loss = model.evaluate(df_val).loss

        results = {
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_loss": val_loss
        }

        with open(os.path.join(self.__log_dir, f"RESUMEN_best_rf_{self.__interval}.json"), "w") as f:
            json.dump(results, f, indent=4)
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            # best_params = {
            #     "num_trees": num_trees,
            #     "max_depth": max_depth,
            #     "min_examples": min_examples,
            # }

        # Save the best model
        self.__model = best_model
        if best_model:
            best_model.to_tensorflow_saved_model(
                self.__best_model_path, mode="tf")
            print(f"Best model saved at {self.__best_model_path}")

        # Evaluate the best model on the validation data
        print(f"Best Validation Accuracy: {best_acc:.4f}")
        self.__cm_file_path = self.plot_confusion_matrix(
            self.__best_model_path,
            y_true=np.concatenate((y, y_val)),
            y_pred=np.concatenate(
                (self.predict(df_train), self.predict(df_val), self.predict(df_test))),
            tags=categories
        )

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              categories: list[str] = None) -> None:
        """Train the Random Forest model without hyperparameter tuning."""

        # Prepare data for YDF
        df_train = pd.DataFrame(X.reshape(X.shape[0], -1))
        df_val = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
        df_train['label'] = np.argmax(y, axis=1)
        df_val['label'] = np.argmax(y_val, axis=1)

        trainer = ydf.RandomForestLearner(label="label")
        self.__model = trainer.train(df_train)

        eval_result = self.__model.evaluate(df_val, name="val")
        acc = eval_result.accuracy
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > self.__best_accuracy:
            self.__best_accuracy = acc
            self.__model.save(self.__best_model_path)
            print(f"New best model saved at {self.__best_model_path}")

        # self.confusion_matrix(
        #     df_val["label"], self.__model.predict(df_val), categories)

    def save_model(self) -> None:
        """Save the trained model."""
        if self.__model is None:
            raise ValueError("Model is not initialized")
        os.makedirs("../models", exist_ok=True)
        self.__model.to_tensorflow_saved_model(
            f"../models/best-rf{self.__interval}.keras")
        self.__best_model_path = f"../models/best-rf{self.__interval}.keras"

    def best_model(self) -> str:
        """Get the best model path.

        Returns:
            str: Path to the best model
        """
        return self.__best_model_path

    def plot_confusion_matrix(self, filename: str, y_true: np.ndarray, y_pred: np.ndarray, tags: list[str]) -> str:
        """Generate and save the confusion matrix."""
        # If y_pred contains probabilities, we convert it to class labels
        # if y_pred.ndim > 1:
        #     y_pred = np.argmax(y_pred, axis=1)

        # Generate confusion matrix
        # self.__model = tf.keras.models.load_model(filename)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        plot_filename = "./RFlog/CM_" + \
            filename.split("/")[-1] + ".png"
        Utils.plot_confusion_matrix(y_true, y_pred, tags, plot_filename,
                                    title=f"Matriz de confusión (Random Forest, intervalo {self.__interval}s)")
        plt.close()
        return plot_filename

    def get_best_acc(self) -> float:
        """Get the best accuracy of the model."""
        return self.__best_accuracy

    def get_confusion_matrix(self) -> str:
        """Get the confusion matrix.

        Returns:
            str: Path to the confusion matrix image
        """
        return self.__cm_file_path

    def stats(self) -> str:
        """Get the stats of the model.

        Returns:
            str: Stats of the model
        """
        return f"Best score: {self.__best_accuracy}"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the trained model."""
        if self.__model is None:
            raise ValueError("Model is not initialized")
        return self.__model.predict(X)

    def get_log_dir(self) -> str:
        """Get the log directory.

        Returns:
            str: Log directory
        """
        return self.__log_dir


if __name__ == "__main__":
    # Generate some dummy tabular data
    X_train = np.random.randn(100, 10)
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(100,)), num_classes=6)

    X_val = np.random.randn(20, 10)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(20,)), num_classes=6)

    X_test = np.random.randn(20, 10)
    y_test = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(20,)), num_classes=6)

    trainer = RandomForestTrainer("10")
    trainer.train_with_hparams(X_train, y_train, X_val, y_val, X_test, y_test, categories=[
        str(i) for i in range(6)], epochs=5)
    print(trainer.stats())
