import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorboard import program
import numpy as np
from RandomForestTrainer import RandomForestTrainer
from RNNTrainer import RNNTrainer
from CNNTrainer import CNNTrainer
from LSTMTrainer import LSTMTrainer
from DataLoader import DataLoader


class Interface:

    def __init__(self):
        self.log_path = None
        self.__models = ["LSTM", "RNN", "CNN", "RandomForest"]
        self.__models_map = {
            "LSTM": LSTMTrainer,
            "RNN": RNNTrainer,
            "CNN": CNNTrainer,
            "RandomForest": RandomForestTrainer
        }
        self.__sec_interval = [0.5]
        self.__blocks = gr.Blocks(theme="ParityError/Interstellar")
        self.__setup_ui()

    def __setup_ui(self):
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
        self.__sec_interval = [0.2, 0.4,
                               0.6, 0.8, 1.0]

    def __change_interval(self, value):
        self.__sec_interval = [value]

    def __train(self, model, path):
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

            label_encoder = LabelEncoder()
            Y_train_encoded = label_encoder.fit_transform(Y_train)
            Y_test_encoded = label_encoder.transform(Y_test)
            Y_val_encoded = label_encoder.transform(Y_val)

            # One-hot encode Y labels
            num_classes = len(label_encoder.classes_)
            Y_train = tf.keras.utils.to_categorical(
                Y_train_encoded, num_classes=num_classes)
            Y_test = tf.keras.utils.to_categorical(
                Y_test_encoded, num_classes=num_classes)
            Y_val = tf.keras.utils.to_categorical(
                Y_val_encoded, num_classes=num_classes)

            trainer.train_with_hparams(
                X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=20, batch_size=1, num_cats=num_classes, categories=label_encoder.classes_)
            trainer.save_model()
            acc = trainer.get_best_acc()
            if acc > best_acc:
                best_acc = acc
                best_interval = interval
                best_model = trainer.best_model()
                cm_file = trainer.get_confusion_matrix()
                stats = trainer.stats()

        return best_model, np.fromfile(cm_file), stats

    def __refresh_tensorboard(self, log_path):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_path])
        url = tb.launch()
        return gr.update()

    def launch(self):
        self.__blocks.launch()


models = ["LSTM", "RNN", "CNN", "RandomForest"]

models_map = {
    "LSTM": LSTMTrainer,
    "RNN": RNNTrainer,
    "CNN": CNNTrainer,
    "RandomForest": RandomForestTrainer
}


sec_interval = [0.5]


def change_interval_all():
    global sec_interval
    sec_interval = [0.2, 0.4,
                    0.6, 0.8, 1.0]


def change_interval(value):
    global sec_interval
    sec_interval = [value]


def train(model, path):
    print(sec_interval)
    best_acc = 0
    best_interval = 0
    best_model = None
    cm_file = None
    stats = None
    for interval in sec_interval:
        trainer = models_map[model](f"{interval}")
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

        label_encoder = LabelEncoder()
        Y_train_encoded = label_encoder.fit_transform(Y_train)
        Y_test_encoded = label_encoder.transform(Y_test)
        Y_val_encoded = label_encoder.transform(Y_val)

        # One-hot encode Y labels
        num_classes = len(label_encoder.classes_)
        Y_train = tf.keras.utils.to_categorical(
            Y_train_encoded, num_classes=num_classes)
        Y_test = tf.keras.utils.to_categorical(
            Y_test_encoded, num_classes=num_classes)
        Y_val = tf.keras.utils.to_categorical(
            Y_val_encoded, num_classes=num_classes)

        trainer.train_with_hparams(
            X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=20, batch_size=1, num_cats=num_classes, categories=label_encoder.classes_)
        trainer.save_model()
        acc = trainer.get_best_acc()
        if acc > best_acc:
            best_acc = acc
            best_interval = interval
            best_model = trainer.best_model()
            cm_file = trainer.get_confusion_matrix()
            stats = trainer.stats()

    return best_model, np.fromfile(cm_file), stats


def refresh_tensorboard(log_path):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_path])
    url = tb.launch()
    return gr.update()


if __name__ == "__main__":
    interface = Interface()
    interface.launch()
