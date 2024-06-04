FROM tensorflow/tensorflow:2.16.1-gpu
WORKDIR /env
RUN apt-get update
RUN apt-get install -y graphviz
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /src
