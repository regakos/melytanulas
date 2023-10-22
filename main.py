import gradio as gr
import pandas as pd

#Read the CoQA dataset
coqa = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')

def display_header():
    return coqa.head().to_string()

#Create a Gradio interface
demo = gr.Interface(fn=display_header, inputs=None, outputs="text", title="CoQA DataFrame Header")

demo.launch()