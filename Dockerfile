FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install build-essential -y
RUN pip install --use-pep517 pyjedai
