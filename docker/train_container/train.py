#import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer, AdamW
import os
#from torch.utils.data import DataLoader, Dataset

# Load DistilBERT model and tokenizer
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')



# Save the fine-tuned model
save_directory = '/workspace/model/'

os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)