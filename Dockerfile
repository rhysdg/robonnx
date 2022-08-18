#FROM python:latest
FROM qts8n/cuda-python:devel

RUN mkdir -p /home/app/
WORKDIR /home/app

COPY setup.py setup.py

COPY . .
ENV CUDA_HOME /usr/local/10.2

RUN cd /home/app && python -m pip install .

CMD ["python", "/home/app/roberta_onnx.py"]
CMD ["bash"]
