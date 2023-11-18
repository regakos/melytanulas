import gradio as gr
import pandas as pd

coqa = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')

def display_header():
    return coqa.head().to_string()

iface = gr.Interface(
    fn=display_header,
    inputs=None,
    outputs=gr.Textbox(),
    title="COQA Dataset Header",
    description="Display the header of the COQA dataset."
)
iface.launch(server_name="0.0.0.0")