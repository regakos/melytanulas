import gradio as gr
import pandas as pd

coqa = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')

def random_response(message, history=None):
    return message

iface = gr.ChatInterface(fn=random_response,
    title="Question answering with Distillbert",
    description="This bot gives an answer for your questions, it doesn't have memory",
    )

iface.launch(server_name="0.0.0.0")