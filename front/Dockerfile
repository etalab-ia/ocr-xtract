FROM python:3.7-buster

COPY requirements.txt requirements.txt

RUN apt-get update \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

COPY . .
WORKDIR .
CMD [ "streamlit", "run", "app.py", "--server.enableCORS=false", "--server.enableXsrfProtection=false","--server.enableWebsocketCompression=false" ]