import os
from flask import Flask, render_template, flash, redirect, url_for, session, request, flash
from .forms import UploadImageForm
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
import pandas as pd
import sys
import base64
import random
import json
import time

UPLOAD_FOLDER = './static/uploaded_image'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

from flask_bootstrap import Bootstrap
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pokemon'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadImageForm()
    if request.method == 'POST':
        data_file = request.files['data_file']
        if data_file and allowed_file(data_file.filename):
            filename = data_file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data_file.save(path)
            session["filename"] = filename
            if "control_id" not in session:
                session["control_id"] = 1
            else:
                session["control_id"] += 1
            return redirect(url_for('result_download'))
    '''
    if form.validate_on_submit():
        file = form.data_file.data
        filename = secure_filename(file.filename)
        session["filename"] = filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    '''
        #flash('Upload success! Please wait while we run our model.', 'alert-success')

    return render_template('home.html', form=form)

@app.route('/result_download', methods=['GET', 'POST'])
def result_download():
    filename = session.get("filename", None)
    control_id = session.get("control_id", None)
    return render_template('result_download.html', filename=filename, control_id=control_id)






def run_predictions(datafiles):
    """
    in: list of datafile
    out: list of data
        [
            {
                "filename": "img1.jpg",
                "image": [[1,1...],[1,1...]...] #img array
                "annotated_image":  [[1,1...],[1,1...]...], #img array
                "prediction": [
                    {
                        'predicted_class': 1,
                        'confidence': 0.538972,
                        'bounding_box': [0.52875,0.23871,0.42894,0.4284]
                    },
                    ...
                ]
            },
            {
                next image
            },
            ...
        ]
    """

    time.sleep(5)

    def temp(df):
        data = {
            "filename": df.filename,
            "image": df.data,
            "annotated_image": df.data,
            "prediction": get_prediction(df.data)
        }
        return data

    return list(map(temp, datafiles))






# RESTAPI
@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.get_json())
    img = base64.b64decode(data['image_base64'].encode('utf8'))
    data['prediction'] = get_prediction(img)

    return json.dumps(data)

def get_prediction(img):
    """
    TODO THIS IS WHERE OUR ENTIRE PREDICTION CODE WILL GO
    """

    preds = [mock_pred() for i in range(3)]
    return preds

def mock_pred():
    return {
        'predicted_class': random.randint(1,4),
        'confidence': random.random(),
        'bounding_box': [random.random(), random.random(), random.random(), random.random()]
    }

@app.route('/testpage')
def test():
    return 'this is a testpage'
