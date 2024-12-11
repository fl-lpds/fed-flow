FROM python:3.10

WORKDIR /fed-flow

ADD requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

ADD . .

CMD ["python3", "main.py"]
