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

def allowed_file(filename): #returns True if is an image
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadImageForm()
    if request.method == 'POST':

        # For single file upload
        #data_file = request.files['data_file']
        # print(data_file)

        #if data_file and allowed_file(data_file.filename):
        #    filename = data_file.filename
        #    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #    data_file.save(path)
        #    session["filename"] = filename

        # For more than one file upload
        all_files = request.files.getlist("data_file")
        print(all_files)

        if not all(allowed_file(x.filename) == True for x in all_files):
            flash('Upload failed! Please ensure to only upload an image file.', 'alert-danger')
            return redirect(url_for('home') +"#contact")

        if all(allowed_file(x.filename) == True for x in all_files):
            for file in all_files:
                filename = file.filename
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
            #flash('Upload success! Please wait while we run our model...', 'alert-success')
            #return redirect(url_for('home') + "#contact")
            #time.sleep(180)
            results = run_predictions(all_files)
            print(results)

            return redirect(url_for('result_download'))
    return render_template('home.html', form=form)

@app.route('/result_download', methods=['GET', 'POST'])
def result_download():
    #full_filename = session.get("filename", None)
    #filename = full_filename.rsplit('.', 1)[0]
    #filetype = full_filename.rsplit('.', 1)[1]
    return render_template('result_download.html')

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
            "image": df,
            "annotated_image": df,
            "prediction": get_prediction(df)
        }
        return data

    return list(map(temp, datafiles))

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
