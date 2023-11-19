# melytanulas
Question answering with DistilBERT


Team name: MMRASG 


Team members' names and Neptun codes: 
Molnár Márk - B44W74;
Regényi Ákos - OWPAZM;
Sáfrán Gergely- FT6QWV; 


Functions of the files in the repository:

Dockerfile for building the docker image

requirements.txt: for defining the needed packages


Project description:
Our project is about using the DistilBERT transformer model for question answering. There are different types of question answering, in our work we give text to the model as input and a related (in-context) question. The model gives the proper answer to the predefined question. We create a user interface to input the data and output the answer. The model is trained on the ... dataset. 


related works (papers, GitHub repositories, blog posts, etc),:

BERT link: https://arxiv.org/abs/1810.04805

DistilBERT link: https://arxiv.org/pdf/1910.01108.pdf

Bert explained: https://medium.com/analytics-vidhya/question-answering-system-with-bert-ebe1130f8def

https://www.kaggle.com/code/arunmohan003/question-answering-using-bert

https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model#:~:text=BERT%2C%20which%20stands%20for%20Bidirectional,calculated%20based%20upon%20their%20connection.

Data source: http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json


How to run it:

docker build -t hw .

docker run -p 7860:7860 --name hw_container hw

How to run pipeline:
1. Build docker image, then run it.
2. Access gradio web interface on the 7860 port. (now its on localhost)
3. Ask the model about a specific context text, which is burnt in right now, then it gives an answer related to the text in a chat box.

Model training is automatic in the script, in the training section, this trains the baseline model with the squad database. 
Evaluation is in the predict_evaluate function.

