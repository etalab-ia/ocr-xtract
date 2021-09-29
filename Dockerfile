FROM python:3.7-buster

COPY requirements.txt requirements.txt
COPY requirements_train.txt requirements_train.txt

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt-get update \
    && apt-get install --no-install-recommends git ffmpeg libsm6 libxext6 poppler-utils -y \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install -r requirements_train.txt \
    && pip cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

RUN python -m spacy download fr_core_news_lg

COPY download_doctr_models.py .
RUN python download_doctr_models.py

COPY . .
WORKDIR .
