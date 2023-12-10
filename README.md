# melytanulas
Question answering with DistilBERT


Team name: MMRASG 


Team members' names and Neptun codes: 
Molnár Márk - B44W74;
Regényi Ákos - OWPAZM;
Sáfrán Gergely- FT6QWV; 


Project description:
Our project is about using the DistilBERT transformer model for question answering. There are different types of question answering, in our work we give text to the model as input and a related (in-context) question. The model gives the proper answer to the predefined question. We create a user interface to input the data and output the answer. The model is trained on the ... dataset. 



Functions of the files in the repository:

-Milestone1.ipynb: The colab file for the project at the time of the first milestone.

-Milestone2.ipynb: The colab file for the project at the time of the second milestone.

-Readme.md (this file for describing the project)


Related works (papers, GitHub repositories, blog posts, etc),:

-BERT link: https://arxiv.org/abs/1810.04805

-DistilBERT link: https://arxiv.org/pdf/1910.01108.pdf

-Bert explained: https://medium.com/analytics-vidhya/question-answering-system-with-bert-ebe1130f8def

-https://www.kaggle.com/code/arunmohan003/question-answering-using-bert

-https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model#:~:text=BERT%2C%20which%20stands%20for%20Bidirectional,calculated%20based%20upon%20their%20connection.

-Data source: http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json


How to run it:

-Use command:
'docker-compose up'
from the docker/inference_container folder to build the docker image and run it.


How to run pipeline:
1. Build docker image and run it. (with the 'docker-compose up' command)
2. Access gradio web interface on the 7860 port. (now its on localhost http://localhost:7860/) 
3. From the dropdown list select a 'topic' (list of words), for the context you are interested in.
4. Ask the model about the specific context, in the chatbox, then it gives an answer related to the text in the same chat box.

Model training is in the train.py file in the docker/train_container folder. This trains the baseline model with the squad database. 
Evaluation is in the predict_answers_and_evaluate function.

