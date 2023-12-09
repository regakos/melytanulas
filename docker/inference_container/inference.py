import gradio as gr
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from transformers import DistilBertForQuestionAnswering
from google.colab import drive

load_directory = '/workspace/model/'

# Construct the full paths for model and tokenizer files

# Load DistilBERT model and tokenizer
#model = DistilBertForQuestionAnswering.from_pretrained(load_directory)

drive.mount('/content/drive')

checkpoint =  "distilbert-base-uncased"

model = DistilBertForQuestionAnswering.from_pretrained(checkpoint)

model.load_state_dict(torch.load('/content/drive/MyDrive/OnlabMSc/modellke.pth'))

tokenizer = DistilBertTokenizer.from_pretrained(load_directory)

#Dataset
paragraph = ''' Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance
                on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or
                decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection
                of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning
                is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods,
                theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory
                data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics. '''

coqa = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')

def ask_model(question,context):

  #question = "Who has the highest cases?"

  encoding = tokenizer.encode_plus(question, context)


  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  start_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))["start_logits"]
  end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))["end_logits"]

  ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
  answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

  answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

  return answer_tokens_to_string

# def ask_model(question, paragraph=paragraph):
#     encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)
#     inputs = encoding['input_ids']  #Token embeddings
#     sentence_embedding = encoding['token_type_ids']  #Segment embeddings
#     tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens
#     start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]),return_dict=False )
#     start_index = torch.argmax(start_scores)
#     end_index = torch.argmax(end_scores)
#     answer = ' '.join(tokens[start_index:end_index+1])
#     corrected_answer = ''

#     for word in answer.split():

#     #If it's a subword token
#       if word[0:2] == '##':
#           corrected_answer += word[2:]
#       else:
#           corrected_answer += ' ' + word
#     return corrected_answer



def response(message, history=None):
    return ask_model(message, paragraph)

iface = gr.ChatInterface(fn=response,
    title="Question answering with Distillbert",
    description="This bot gives an answer for your questions, it doesn't have memory",
    )

iface.launch(server_name="0.0.0.0")
