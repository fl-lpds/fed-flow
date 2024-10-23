FROM autlpdslab/fedflow:base
COPY requirements4.txt /fed-flow/
RUN pip install --upgrade setuptools
RUN pip install -r /fed-flow/requirements4.txt
COPY app /fed-flow/app
COPY energy-estimation /fed-flow/energy
WORKDIR /fed-flow/
ENV PYTHONPATH=/fed-flow
