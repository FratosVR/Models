import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorboard import program
import numpy as np
from RandomForestTrainer import RandomForestTrainer
from RNNTrainer import RNNTrainer
from CNNTrainer import CNNTrainer
from LSTMTrainer import LSTMTrainer
from DataLoader import DataLoader


class Interface:
    """
    Interface for training models with Gradio and TensorBoard integration.
    """

    def __init__(self):
        """
        Initializes the interface with the necessary components and settings.
        """

        """ Log path for TensorBoard """
        self.log_path: str = None
        """ List of available models for training """
        self.__models: list[str] = ["LSTM", "RNN", "CNN", "RandomForest"]
        """ Mapping of model names to their respective trainer classes """
        self.__models_map: dict[str, LSTMTrainer | RNNTrainer | CNNTrainer | RandomForestTrainer] = {
            "LSTM": LSTMTrainer,
            "RNN": RNNTrainer,
            "CNN": CNNTrainer,
            "RandomForest": RandomForestTrainer
        }
        """ Interval for training in seconds """
        self.__sec_interval = [0.5]
        """ Gradio Blocks for the user interface """
        self.__blocks = gr.Blocks(theme="ParityError/Interstellar")
        """ Initialize the user interface """
        self.__setup_ui()

    def __setup_ui(self):
        """
        Sets up the user interface components using Gradio Blocks.
        """
        with self.__blocks:
            gr.Markdown("# Model trainer")
            with gr.Tab("Train"):
                with gr.Row():
                    with gr.Column():
                        model_dd = gr.Dropdown(
                            self.__models, label="Select model")
                        path = gr.Textbox(label="Path to dataset")
                        slider = gr.Slider(0.1, 1.0, value=0.5,
                                           step=0.1, label="Sec interval")
                        slider.change(self.__change_interval,
                                      inputs=slider, outputs=None)
                        all_interval_button = gr.Button("All intervals")
                        button = gr.Button("Train model")

                    with gr.Column():
                        trained_model = gr.File(label="Trained model")
                        conf_mat = gr.Image(
                            label="Confusion matrix", type="filepath")
                        stats = gr.Textbox(label="Training statistics")

                        all_interval_button.click(
                            self.__change_interval_all, inputs=None, outputs=None)
                    button.click(self.__train, inputs=[model_dd, path], outputs=[
                                 trained_model, conf_mat, stats])
            with gr.Tab("Tensorboard"):
                log_path = gr.Textbox(
                    label="Log path", value="./logs", interactive=True)
                log_buttom = gr.Button("Refresh tensorboard")
                iframe = gr.HTML(f"""
<iframe src="http://localhost:6006" width="100%" height="800px" frameborder="0"></iframe>
""")
                log_buttom.click(self.__refresh_tensorboard,
                                 inputs=[log_path], outputs=[iframe])

    def __change_interval_all(self):
        """
        Changes the interval for all models to a predefined set of values.
        """
        self.__sec_interval = [0.2, 0.4,
                               0.6, 0.8, 1.0]

    def __change_interval(self, value):
        """
        Changes the interval for the selected model to a single value.
        :param value: The new interval value to set.
        """
        self.__sec_interval = [value]

    def __train(self, model, path):
        """
        Trains the selected model with the specified dataset path and interval.
        :param model: The model to train.
        :param path: The path to the dataset.
        :return: The best trained model, confusion matrix, and training statistics.
        """
        print(self.__sec_interval)
        best_acc = 0
        best_interval = 0
        best_model = None
        cm_file = None
        stats = None
        for interval in self.__sec_interval:
            trainer = self.__models_map[model](f"{interval}")
            global log_path
            trainer.get_log_dir()
            dl = DataLoader(
                path, interval)
            data = dl.load_dataset()
            train_data, test_data = train_test_split(
                data, test_size=0.40, random_state=42)
            test_data, val_data = train_test_split(
                test_data, test_size=0.50, random_state=42)
            Y_train, X_train = zip(*train_data)
            Y_test, X_test = zip(*test_data)
            Y_val, X_val = zip(*val_data)

            X_train = np.array(X_train, dtype=np.float32)

            X_test = np.array(X_test, dtype=np.float32)

            X_val = np.array(X_val, dtype=np.float32)

            label_encoder = OneHotEncoder()
            Y_train_encoded = label_encoder.fit_transform(
                np.array(Y_train).reshape(-1, 1)).toarray()
            Y_test_encoded = label_encoder.transform(
                np.array(Y_test).reshape(-1, 1)).toarray()
            Y_val_encoded = label_encoder.transform(
                np.array(Y_val).reshape(-1, 1)).toarray()

            Y_train = np.array(Y_train_encoded, dtype=np.float32)
            Y_test = np.array(Y_test_encoded, dtype=np.float32)
            Y_val = np.array(Y_val_encoded, dtype=np.float32)

            categories = label_encoder.categories_[0]
            trainer.train_with_hparams(
                X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=20, batch_size=1, num_cats=len(categories), categories=categories)
            trainer.save_model()
            acc = trainer.get_best_acc()
            if acc > best_acc:
                best_acc = acc
                best_interval = interval
                best_model = trainer.best_model()
                cm_file = trainer.get_confusion_matrix()
                stats = trainer.stats()

        return best_model, np.fromfile(cm_file), stats

    def __refresh_tensorboard(self, log_path: str) -> gr.update:
        """ Refreshes the TensorBoard interface with the specified log path.

        Args:
            log_path (str): The path to the TensorBoard logs.

        Returns:
            gr.update(): An update to the Gradio interface to reflect the new TensorBoard state.
        """
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_path])
        url = tb.launch()
        return gr.update()

    def launch(self):
        """
        Launches the Gradio interface and starts the TensorBoard server.
        """
        self.__blocks.launch()


if __name__ == "__main__":
    interface = Interface()
    interface.launch()
