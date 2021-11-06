import os
import base64
import random
import json
import time
import numpy as np
import cv2
from flask import Flask, render_template, redirect, url_for, session, request, flash
from flaskapp.forms import UploadImageForm, PreviewImageForm
from datetime import datetime
from flaskapp.model import mask_img, run_model
from flask_bootstrap import Bootstrap

UPLOAD_FOLDER = 'flaskapp/static/uploaded_image'
PREVIEW_FOLDER = 'flaskapp/static/preview'
OUTPUT_FOLDER = 'flaskapp/static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pokemon'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
Bootstrap(app)

def allowed_file(filename): #returns True if is an image
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS #TODO should [-1] instead? potential bug if filename got . right?

@app.route('/', methods=['GET', 'POST'])
def home():
    form1 = UploadImageForm()
    form2 = PreviewImageForm()

    if request.method == 'POST' and form2.validate_on_submit():
        image = request.files['data_file_preview']
        image_name = image.filename

        ## Saving original image
        path = os.path.join(app.config['PREVIEW_FOLDER'], image_name)
        image.save(path)

        ## Masking Process
        to_be_masked_image = cv2.imread(path)
        masked_image = mask_img(to_be_masked_image) # use .shape
        masked_image_name = 'masked_' + image.filename
        image_size1= masked_image.shape[0] / 10
        image_size2 = masked_image.shape[1] / 10

        ## Saving masked image
        cv2.imwrite(os.path.join(app.config['PREVIEW_FOLDER'], masked_image_name), masked_image)
        return render_template('home.html', form1=form1, form2=form2, image_name=image_name, masked_image_name=masked_image_name, image_size1=image_size1, image_size2=image_size2)

    if request.method == 'POST' and 'mask' in request.form:
        all_files = request.files.getlist("data_file")
        if not all(allowed_file(x.filename) == True for x in all_files):
            flash('Upload failed! Please ensure to only upload an image file.', 'alert-danger')
            return redirect(url_for('home') +"#contact")

        if all(allowed_file(x.filename) == True for x in all_files):
            unique_num = str(str(datetime.today()).split(".")[0])[-9:].replace(':', '')[1:]
            session['unique_num'] = unique_num
            print("Unique Number: ", unique_num)
            print(len(unique_num))

            for file in all_files:
                filename = file.filename
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

            mask = form1.mask.data  ## mask refers to the boolean of either true or false
            print("Masking =", mask)

            results = run_predictions(all_files, mask)
            print(results)
            name_of_result = unique_num + '.json'
            os.chdir(app.config['OUTPUT_FOLDER'])

            with open(name_of_result, 'w') as f:
                json.dump(results, f)

            return redirect(url_for('result_download'))
    return render_template('home.html', form1=form1, form2=form2)

@app.route('/result_download', methods=['GET', 'POST'])
def result_download():
    unique_num = session.get("unique_num", None)
    return render_template('result_download.html', unique_num=unique_num)

def run_predictions(datafiles, mask=True):
    
    # can transform ur datafiles to correct format here




    #######################################################




    """
    INPUT of run_model()
    - run_model(data, mask=True) where data is list of dict -> [{}, {}, {}]
    data's each dict requires 2 keys
    {
        'filename': filename eg '1.jpg',
        'img': image in array format eg. output of cv2.imread()
    }


    OUTPUT
    output_json: list of data
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

    file_df: output_json but in df format
    final_counts: df of final counts
    """
    # output_json, file_df, final_counts = run_model(data, mask = True) # dir = directory to file containing images



    def temp(df):
        data = {
            "filename": df.filename,
            #"image": df,
            #"annotated_image": df,
            "predictions": get_prediction(df)
        }
        return data

    return list(map(temp, datafiles))

def tobase64(img):
    return cv2.imencode('.jpg', img)[1].tobytes()

def get_prediction(img):

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
    start = time.time()
    data = json.loads(request.get_json())
    img_binary = base64.b64decode(data['image_base64'].encode('utf8'))
    img = cv2.imdecode(np.frombuffer(img_binary, np.uint8), flags=1)

    d = {
        'filename': data['filename'],
        'img': img
    }

    output_json, file_df, final_counts = run_model([d], mask = True)

    data['prediction'] = output_json[0]['predictions']
    data['image_base64'] = data['image_base64'][:20] # TODO TEMP TRUNCATE, REMOVE LATER

    print(time.time() - start)
    return json.dumps(data)

@app.route('/testpage')
def test():
    return 'this is a testpage'


if __name__ == '__main__':
    app.run(host='localhost', port=5000)