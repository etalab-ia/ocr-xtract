FROM python
RUN apt-get update -y
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get -y install tesseract-ocr
COPY . .
WORKDIR .
RUN pip install pillow
RUN pip install pytesseract
RUN pip install opencv-contrib-python
RUN pip install -r requirements.txt
WORKDIR ./api
ENTRYPOINT ["python"]
CMD ["app.py"]