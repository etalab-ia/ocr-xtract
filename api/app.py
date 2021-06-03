import json
import logging
import os

from pathlib import Path
from werkzeug.utils import secure_filename
from flask_restful import Api, Resource, reqparse, fields, marshal
from flask import Flask
from flask import request, jsonify

from src.document.CNI import CNI

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)

UPLOAD_DIRECTORY = Path("./api/api_uploaded_files")
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


class OCRRectoCNI (Resource):
    def __init__(self, **kwargs):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('files')
        self.ALLOWED_EXTENSIONS = {'pdf','jpeg','png','jpg'}
        self.logger = kwargs.get('logger')

    def get(self):
        # self.logger - 'logger' from resource_class_kwargs
        return self.logger.name

    def allowed_file(self, filename):
        # this has changed from the original example because the original did not work for me
        return filename[-3:].lower() in self.ALLOWED_EXTENSIONS

    def post(self):
        data = {"success": False}

        """Upload a file."""
        file = request.files['file'] #get file in the request
        self.logger.debug(self.allowed_file(file.filename))
        if file and self.allowed_file(file.filename):
            filename = secure_filename(file.filename) #make sure we have a proper filename
            self.logger.debug(f'**found {filename}')
            full_filename = UPLOAD_DIRECTORY / filename
            file.save(full_filename) #saves file in folder
            cni = CNI(recto_path=full_filename)
            cni.align_images()
            cni.clean_images()
            cni.extract_ocr()
            results = cni.export_ocr()
            # files are removed after use
            os.remove(full_filename)
            data["result"] = results
            data["success"] = True
        return data


api.add_resource(OCRRectoCNI, '/', resource_class_kwargs={
    # any logger here...
    'logger': logging.getLogger('my_custom_logger')
})

if __name__ == "__main__":
    app.run(debug=True, port=8000)