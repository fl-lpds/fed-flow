FROM python:3.10

WORKDIR /fed-flow

ADD requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

ADD . .

WORKDIR /fed-flow/app/fl_training/runner
CMD ["python3", "fed_base_run.py"]
