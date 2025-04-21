import os
import numpy as np
import pandas as pd
import time
import ydf
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import TensorBoard


class RandomForestTrainer:
    """Random Forest trainer using Yggdrasil Decision Forests (ydf) with manual hyperparameter tuning and TensorBoard logging."""

    def __init__(self, interval: str, model_dir: str = "./models", log_dir: str = "./logs") -> None:
        self.__interval = interval
        self.__model = None
        self.__best_model_path = os.path.join(model_dir, f"best_rf_{interval}")
        self.__model_dir = model_dir
        self.__log_dir = log_dir
        self.__best_accuracy = 0.0
        self.__tensorboard_callbacks = [TensorBoard(
            log_dir=os.path.join(log_dir, f"rf_{interval}"))]

    def train_with_hparams(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                           categories: list[str] = None, epochs: int = 10) -> None:
        """Manually search for the best hyperparameters for the Random Forest model."""

        # Define hyperparameter grid
        num_trees_options = [50, 100, 150, 200]
        max_depth_options = [3, 5, 10, 15]
        min_examples_options = [2, 5, 8, 10]

        best_acc = 0.0
        best_model = None
        best_params = {}

        for num_trees in num_trees_options:
            for max_depth in max_depth_options:
                for min_examples in min_examples_options:
                    print(
                        f"Training model with num_trees={num_trees}, max_depth={max_depth}, min_examples={min_examples}")

                    # Prepare data for YDF (convert to pandas DataFrame)
                    df_train = pd.DataFrame(
                        X.reshape(X.shape[0], -1), columns=[f"feature_{i}" for i in range(X.shape[1])])
                    df_val = pd.DataFrame(X_val.reshape(
                        X_val.shape[0], -1), columns=[f"feature_{i}" for i in range(X_val.shape[1])])

                    # Ensure 'label' column is added and is of string type
                    df_train['label'] = np.argmax(y, axis=1)
                    df_val['label'] = np.argmax(y_val, axis=1)

                    # Create and train the Random Forest model
                    learner = ydf.RandomForestLearner(
                        label="label",
                        num_trees=num_trees,
                        max_depth=max_depth,
                        min_examples=min_examples,
                    )

                    model = learner.train(df_train)
                    eval_result = model.evaluate(df_val)
                    acc = eval_result.accuracy
                    print(f"Validation Accuracy: {acc:.4f}")

                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
                        best_params = {
                            "num_trees": num_trees,
                            "max_depth": max_depth,
                            "min_examples": min_examples,
                        }

        # Save the best model
        self.__model = best_model
        if best_model:
            best_model.save(self.__best_model_path)
            print(f"Best model saved at {self.__best_model_path}")

        # Evaluate the best model on the validation data
        print(f"Best Validation Accuracy: {best_acc:.4f}")
        self.confusion_matrix(
            df_val["label"], best_model.predict(df_val), categories)

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

        self.confusion_matrix(
            df_val["label"], self.__model.predict(df_val), categories)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if self.__model is None:
            raise ValueError("Model is not initialized")
        df = pd.DataFrame(X.reshape(X.shape[0], -1))
        return self.__model.predict(df)

    def confusion_matrix(self, y_true, y_pred, labels=None) -> str:
        """Generate and save the confusion matrix."""
        # If y_pred contains probabilities, we convert it to class labels
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Reds, xticks_rotation=45)

        os.makedirs("./RFlog", exist_ok=True)
        filename = f"./RFlog/CM_rf_{self.__interval}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Confusion matrix saved to {filename}")
        return filename

    def best_model(self) -> str:
        """Get path to best model."""
        return self.__best_model_path

    def stats(self) -> str:
        """Get the stats of the best model."""
        return f"Best score: {self.__best_accuracy:.4f}"

    def load_best_model(self) -> None:
        """Load the best saved model."""
        self.__model = ydf.load_model(self.__best_model_path)


if __name__ == "__main__":
    # Generate some dummy tabular data
    X_train = np.random.randn(100, 10)
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(100,)), num_classes=6)

    X_val = np.random.randn(20, 10)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 6, size=(20,)), num_classes=6)

    trainer = RandomForestTrainer("10")
    trainer.manual_hyperparam_search(X_train, y_train, X_val, y_val, categories=[
        str(i) for i in range(6)], epochs=5)
    print(trainer.stats())
