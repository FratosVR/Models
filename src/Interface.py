import gradio as gr
from gradio_iframe import iFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorboard import program
import numpy as np
# from RandomForestTrainer import RandomForestTrainer
# from RNNTrainer import RNNTrainer
# from CNNTrainer import CNNTrainer
from LSTMTrainer import LSTMTrainer
from DataLoader import DataLoader


models = ["LSTM"]

models_map = {
    "LSTM": LSTMTrainer,
}

log_path = "./logs"
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_path])
url = tb.launch()


sec_interval = [0.5]


def change_interval_all():
    sec_interval = [0.1, 0.2, 0.3, 0.4, 0.5,
                    0.6, 0.7, 0.8, 0.9, 1.0]


def change_interval(value):
    sec_interval = [value]


def train(model, path):
    for interval in sec_interval:
        trainer = models_map[model](interval)
        dl = DataLoader(
            path, interval)
        data = dl.load_dataset()
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42)
        Y_train, X_train = zip(*train_data)
        Y_test, X_test = zip(*test_data)

        X_train = np.array(X_train, dtype=np.float32)

        X_test = np.array(X_test, dtype=np.float32)

        label_encoder = LabelEncoder()
        Y_train_encoded = label_encoder.fit_transform(Y_train)
        Y_test_encoded = label_encoder.transform(Y_test)

        # One-hot encode Y labels
        num_classes = len(label_encoder.classes_)
        Y_train = tf.keras.utils.to_categorical(
            Y_train_encoded, num_classes=num_classes)
        Y_test = tf.keras.utils.to_categorical(
            Y_test_encoded, num_classes=num_classes)

        trainer.train_with_hparams(
            X_train, Y_train, X_test, Y_test, epochs=5, batch_size=2, num_cats=num_classes)
        trainer.save_model()
        trainer.predict(X_test[0])

        return trainer.best_model(), NULL, trainer.stats()


with gr.Blocks(theme="ParityError/Interstellar") as blocks:
    gr.Markdown("# Model trainer")
    with gr.Tab("Train"):
        with gr.Row():
            with gr.Column():
                model_dd = gr.Dropdown(models, label="Select model")
                path = gr.Textbox(label="Path to dataset")
                slider = gr.Slider(0.1, 1.0, value=0.5,
                                   step=0.1, label="Sec interval")
                slider.release(change_interval, inputs=slider, outputs=None)
                all_interval_button = gr.Button("All intervals")
                button = gr.Button("Train model")

            with gr.Column():
                trained_model = gr.File(label="Trained model")
                conf_mat = gr.Image(label="Confusion matrix")
                stats = gr.Textbox(label="Training statistics")

                all_interval_button.click(
                    change_interval_all, inputs=None, outputs=None)
            button.click(train, inputs=[model_dd, path], outputs=[
                trained_model, conf_mat, stats])
    with gr.Tab("Tensorboard"):
        gr.HTML(f"""
<iframe src="{url}" width="100%" height="800px" frameborder="0"></iframe>
""")

print("Tensorboard at ", url)
blocks.launch()
