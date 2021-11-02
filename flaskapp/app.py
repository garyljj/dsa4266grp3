import os
import base64
import random
import json
import time
import numpy as np
import cv2
from flask import Flask, render_template, redirect, url_for, session, request, flash
from .forms import UploadImageForm, PreviewImageForm
from datetime import datetime
from .model import mask_img
from flask_bootstrap import Bootstrap

UPLOAD_FOLDER = 'flaskapp/static/uploaded_image'
PREVIEW_FOLDER = 'flaskapp/static/preview'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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
    form1 = UploadImageForm()
    form2 = PreviewImageForm()

    if request.method == 'POST' and form2.validate_on_submit():
        image = request.files['data_file_preview']
        masked_image_name = 'masked_' + image.filename
        npimg = np.fromfile(image, np.uint8)
        to_be_masked_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        masked_image = mask_img(to_be_masked_image)
        cv2.imwrite(os.path.join(app.config['PREVIEW_FOLDER'], masked_image_name), masked_image)

        #original_image_name = image.filename
        #image.save(os.path.join(app.config['PREVIEW_FOLDER'], original_image_name))
        #return render_template('home.html', form1=form1, form2=form2, original_image_name=original_image_name, masked_image_name=masked_image_name)
        return render_template('home.html', form1=form1, form2=form2, masked_image_name=masked_image_name)

    if request.method == 'POST' and 'mask' in request.form:
        all_files = request.files.getlist("data_file")
        if not all(allowed_file(x.filename) == True for x in all_files):
            flash('Upload failed! Please ensure to only upload an image file.', 'alert-danger')
            return redirect(url_for('home') +"#contact")

        if all(allowed_file(x.filename) == True for x in all_files):
            unique_num = str(str(datetime.today()).split(".")[0])[-9:].replace(':', '')
            session['unique_num'] = unique_num
            print("Unique Number: ", unique_num)

            for file in all_files:
                filename = file.filename
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

            mask = form1.mask.data  ## mask refers to the boolean of either true or false
            print("Masking =", mask)
            results = run_predictions(all_files, mask)
            print(results)
            return redirect(url_for('result_download'))
    return render_template('home.html', form1=form1, form2=form2)

@app.route('/result_download', methods=['GET', 'POST'])
def result_download():
    unique_num = session.get("unique_num", None)
    return render_template('result_download.html', unique_num=unique_num)

def run_predictions(datafiles, mask=True):
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
