FROM python:3.10

WORKDIR /fed-flow

RUN --mount=type=cache,target=/root/.cache/pip pip install uv
ADD requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -r requirements.txt

ADD . .

CMD ["python3", "main.py"]
