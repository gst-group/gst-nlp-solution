# Demo for Flask
# jasper.qiu@guanshantech.com

import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename

CURRENT_FILE_PATH = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(CURRENT_FILE_PATH,'temp')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


@app.route('/hello')
def hello_world():
    return 'Hello GST!'


@app.route('/run')
def welcome():
    return 'Welcome to the great company GST!'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return "SUCCESS"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''



if __name__ == '__main__':
    app.run(port=8000)


