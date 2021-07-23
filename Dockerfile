FROM python:3.8.1-slim

COPY requirements.txt requirements.txt

RUN apt-get update \
    && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 -y \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

COPY . .
WORKDIR .
CMD [ "python", "-u" ,"-m", "streamlit.cli", "run", "front/app_local.py" ]