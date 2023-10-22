FROM python:3.9

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

EXPOSE ${GRADIO_SERVER_PORT}

WORKDIR /workspace

ADD requirements.txt main.py /workspace/

RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

CMD ["python", "/workspace/main.py"]