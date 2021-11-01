import os
from flask import Flask, render_template, flash, redirect, url_for, session, request, flash
from .forms import UploadImageForm, PreviewImageForm
import pandas as pd
import sys
import base64
import random
import json
import time
from datetime import datetime
from .model import mask_img
import numpy as np
import cv2

UPLOAD_FOLDER = './static/uploaded_image'
PREVIEW_FOLDER = './static/preview'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

from flask_bootstrap import Bootstrap
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pokemon'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER
Bootstrap(app)

def allowed_file(filename): #returns True if is an image
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    print("HERE")
    form2 = PreviewImageForm()
    form1 = UploadImageForm()

    if form2.validate_on_submit():
        print("HERE2")
        image = request.files['data_file_preview']
        image_name = image.filename

        npimg = np.fromfile(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        masked_image = mask_img(image)
        cv2.imwrite(os.path.join(app.config['PREVIEW_FOLDER'], image_name), masked_image)

        return render_template('home.html', form1=form1, form2=form2, image=masked_image, image_name=image_name)

    if request.method == 'POST':
        # For more than one file upload
        all_files = request.files.getlist("data_file")
        print(all_files)

        if not all(allowed_file(x.filename) == True for x in all_files):
            flash('Upload failed! Please ensure to only upload an image file.', 'alert-danger')
            return redirect(url_for('home') +"#contact")

        if all(allowed_file(x.filename) == True for x in all_files):
            unique_num = str(str(datetime.today()).split(".")[0])[-9:].replace(':', '')
            session['unique_num'] = unique_num
            print(unique_num)
            for file in all_files:
                filename = file.filename
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

            mask = form1.mask.data  ## mask refers to the boolean of either true or false
            print(mask)
            #results = run_predictions(all_files, mask)
            results = run_predictions(all_files)
            print(results)
            #return render_template('result_download.html', unique_num=unique_num)
            return redirect(url_for('result_download'))
    return render_template('home.html', form1=form1, form2=form2)

@app.route('/result_download', methods=['GET', 'POST'])
def result_download():
    unique_num = session.get("unique_num", None)
    return render_template('result_download.html', unique_num=unique_num)

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
