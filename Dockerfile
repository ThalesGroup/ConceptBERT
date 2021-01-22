FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

USER root

RUN pip install --upgrade pip
RUN apt-get update && apt-get upgrade -y
RUN ["apt-get","-y", "install","libcap-dev", "python3-dev"]
ADD requirements.txt requirements.txt
RUN echo "pip3 install requirements.txt"
RUN pip3 install -r requirements.txt

ENV LANG C.UTF-8

ADD kilbert kilbert
RUN cd kilbert/tools/refer && make

