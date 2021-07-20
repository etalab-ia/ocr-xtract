FROM python
RUN apt-get update -y
RUN apt update && apt install -y ffmpeg libsm6 libxext6
RUN apt-get -y install tesseract-ocr
COPY . .
WORKDIR .
RUN pip install -r requirements.txt
CMD [ "python", "-m", "streamlit.cli", "run", "front/app_local.py" ]