FROM python:3.9

WORKDIR /workspace

ADD requirements.txt train.py /workspace/

RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

CMD ["python", "/workspace/train.py"]