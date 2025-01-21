import gradio as gr
from RandomForestTrainer import RandomForestTrainer
from RNNTrainer import RNNTrainer
from CNNTrainer import CNNTrainer
from LSTMTrainer import LSTMTrainer


models = ["RNN", "CNN", "LSTM", "RandomForest"]

models_map = {
    "RNN": RNNTrainer,
    "CNN": CNNTrainer,
    "LSTM": LSTMTrainer,
    "RandomForest": RandomForestTrainer
}


def train(model, path):
    trainer = models_map[model]()
    trainer.train(path)
    return NULL, NULL, NULL


with gr.Blocks(theme="ParityError/Interstellar") as blocks:
    gr.Markdown("# Model trainer")
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(models, label="Select model")
            path = gr.Textbox(label="Path to dataset")
            button = gr.Button("Train model")

        with gr.Column():
            trained_model = gr.File(label="Trained model")
            conf_mat = gr.Image(label="Confusion matrix")
            stats = gr.Textbox(label="Training statistics")
        button.click(train, inputs=[model, path], outputs=[
            trained_model, conf_mat, stats])

blocks.launch()
