import logging
import shutil
import os
from pathlib import Path
from tempfile import mkdtemp

from werkzeug.utils import secure_filename
from flask_restful import Api, Resource, reqparse, fields, marshal
from flask import Flask
from flask import request, jsonify

from src.image.image import Image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


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

            temp_folder = mkdtemp()
            full_filename = os.path.join(temp_folder, filename)
            file.save(full_filename) #saves file in folder

            image = Image(full_filename, 'data_test/cni_recto')
            shutil.rmtree(temp_folder)

            image.align_images()

            data["result"] = image.extract_information()
            data["success"] = True
        return data


class OCRFeuilleDePaye (Resource):
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

            temp_folder = mkdtemp()
            full_filename = os.path.join(temp_folder, filename)
            file.save(full_filename) #saves file in folder

            image = Image(full_filename, 'data_test/salary')
            shutil.rmtree(temp_folder)

            image.align_images()

            data["result"] = image.extract_information()
            data["success"] = True
        return data


api.add_resource(OCRRectoCNI, '/cni', resource_class_kwargs={
    # any logger here...
    'logger': logging.getLogger('my_custom_logger')
})
api.add_resource(OCRFeuilleDePaye, '/fdp', resource_class_kwargs={
    # any logger here...
    'logger': logging.getLogger('my_custom_logger')
})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")