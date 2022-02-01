#FROM python:3.7-buster
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt-get update \
    && apt-get install -y software-properties-common  \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install --no-install-recommends -y git ffmpeg libsm6 libxext6 poppler-utils build-essential  \
        python3-pip \
        python3 \
        python3-dev \
        python3-distutils \
        python3-setuptools \
    && pip3 install --upgrade pip setuptools wheel

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt \
    && pip3 cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

RUN python -m spacy download fr_core_news_lg

COPY download_doctr_models.py .
RUN python download_doctr_models.py

COPY . .
WORKDIR .
