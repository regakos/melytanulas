#Importok
import gradio as gr
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from datasets import Dataset, DatasetDict
import random

#Alap (baseline) modell betöltése
checkpoint =  "distilbert-base-uncased"
model = DistilBertForQuestionAnswering.from_pretrained(checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#Trainelt model betöltése
model.load_state_dict(torch.load('/workspace/model/model.pth'))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
#model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

coqa = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')

df= coqa.copy()

datasets = {"train": []}


for idx, row in df.iterrows():
    context = row['data']['story']
    questions = [q['input_text'] for q in row['data']['questions']]
    answers = row['data']['answers']
    file = row['data']['name']

    for i, (question, answer) in enumerate(zip(questions, answers)):
        flattened_answer = {
            'span_start': answer['span_start'],
            'span_end': answer['span_end'],
            'span_text': answer['span_text']
        }

        # Create an example for each question
        example_data = {
            "id": f"{idx}_{i}",
            "title": f"Example {idx}_{i}",
            "context": context,
            "question": question,
            "answers": flattened_answer,  # Single answer for the question
            "file": file
        }

        datasets["train"].append(example_data)

# Create a DatasetDict
coqa_dataset_dict = DatasetDict({"train": datasets["train"]})

dfoutput = pd.read_excel('/workspace/data/output1.xlsx')
dfoutput.drop(columns=['Unnamed: 0'], inplace = True)
dfoutput

dftopikok = pd.read_excel('/workspace/data/topikok.xlsx')
dftopikok.drop(columns=['Unnamed: 0'], inplace = True)
dftopikok

concatenated_rows = dftopikok.apply(lambda row: ', '.join(row), axis=1)
print(concatenated_rows.tolist())
topicsitems = concatenated_rows.tolist()

selected_item = 'mrs, house, chapter, night, place, mr, room, said, says, think'
index_of_item = topicsitems.index(selected_item)

listofindexes = dfoutput[dfoutput.iloc[:, 0] == index_of_item - 1].index.tolist()

random_index = random.choice(listofindexes)

context_value = datasets["train"][random_index]['context']

updated_context = context_value
default = context_value

def ask_model(question, context):

  encoding = tokenizer.encode_plus(question, context.value)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
  start_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))["start_logits"]
  end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))["end_logits"]

  ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
  answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

  answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
  return answer_tokens_to_string

def update_context(selected_item):

    index_of_item = topicsitems.index(selected_item)
    listofindexes = dfoutput[dfoutput.iloc[:, 0] == index_of_item - 1].index.tolist()
    random_index = random.choice(listofindexes)
    context_value = datasets["train"][random_index]['context']
    updated_context = context_value
    context.value = context_value
    return updated_context


def response(message, history=None):
    return ask_model(message, context)

with gr.Blocks() as demo:
    with gr.Row():
        # Column for inputs
        with gr.Column():
            chat= gr.ChatInterface(fn=response,)



        # Column for outputs
        with gr.Column():
            hide = gr.Dropdown(choices=topicsitems, interactive=True, value="mrs, house, chapter, night, place, mr, room, said, says, think")

            context = gr.Textbox(label="Context", value= default, interactive=False, visible=True)
            hide.change(fn = update_context, inputs=hide, outputs=context)

# Launch the interface
demo.launch(server_name="0.0.0.0")