FROM python:3.8.1-slim
RUN apt-get update -y
RUN apt update && apt install -y ffmpeg libsm6 libxext6
RUN apt-get -y install tesseract-ocr
COPY . .
WORKDIR .
RUN rm -r data/
RUN pip install -r requirements.txt
CMD [ "python", "-u" ,"-m", "streamlit.cli", "run", "front/app_local.py" ]