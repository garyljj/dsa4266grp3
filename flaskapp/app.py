import os
import base64
import random
import json
import time
import numpy as np
import cv2
import shutil
from zipfile import ZipFile
from flask import Flask, render_template, redirect, url_for, session, request, flash
from flaskapp.forms import UploadImageForm, PreviewImageForm
from datetime import datetime
from flaskapp.model import mask_img, run_model
from flask_bootstrap import Bootstrap

UPLOAD_FOLDER = 'flaskapp/static/uploaded_image'
PREVIEW_FOLDER = 'flaskapp/static/preview'
OUTPUT_FOLDER = 'flaskapp/static/output'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pokemon'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
Bootstrap(app)

def allowed_file(filename): #returns True if is an image
    return '.' in filename and \
           filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    form1 = UploadImageForm()
    form2 = PreviewImageForm()

    ## Preview of Masking Code
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

    ## Prediction Code
    if request.method == 'POST' and 'mask' in request.form:
        all_files = request.files.getlist("data_file")

        ## If not all files are images
        if not all(allowed_file(x.filename) == True for x in all_files):
            flash('Upload failed! Please ensure to only upload an image file.', 'alert-danger')
            return redirect(url_for('home') +"#contact")

        ## If all files are images
        if all(allowed_file(x.filename) == True for x in all_files):
            unique_num = str(str(datetime.today()).split(".")[0])[-9:].replace(':', '')[1:] ## Creating a unique_num for tracking
            session['unique_num'] = unique_num
            print("Unique Number: ", unique_num)

            all_names = [] ## all_names: List to hold the names of the files
            for file in all_files:
                filename = file.filename
                all_names.append(filename[:-4]) ## Assuming all images end with .jpg
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

            ## Status of Masking:
            mask = form1.mask.data  ## mask refers to the boolean of either true or false
            print("Masking =", mask)

            ## Prediction using run_predictions
            output_json, a_img, final_counts = run_predictions(all_files, mask)

            ## Looping through to name and output the json and image, and then adding to the zip file
            zip_name = 'run_' + unique_num + '.zip'
            with ZipFile(os.path.join(app.config['OUTPUT_FOLDER'], zip_name), 'w') as z:
                for output, img, name in zip(output_json, a_img, all_names):
                    json_name = name + '.json'
                    image_name = name + '.jpg'
                    with open(os.path.join(app.config['OUTPUT_FOLDER'], json_name), 'w') as f:
                        json.dump(output, f)
                    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], image_name), img)
                    z.write(os.path.join(app.config['OUTPUT_FOLDER'], json_name), 'outputs/' + json_name)
                    z.write(os.path.join(app.config['OUTPUT_FOLDER'], image_name), 'outputs/' + image_name)

            ## Zipping everything in the output folder
            #shutil.make_archive(unique_num, 'zip', OUTPUT_FOLDER)

            return redirect(url_for('result_download'))
    return render_template('home.html', form1=form1, form2=form2)

@app.route('/result_download', methods=['GET', 'POST'])
def result_download():
    unique_num = session.get("unique_num", None)
    return render_template('result_download.html', unique_num=unique_num)

def run_predictions(datafiles, mask=True):

    d = []
    for datafile in datafiles:
        datafile_name = datafile.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], datafile_name)
        transformed_image = cv2.imread(path)
        d.append({'filename': datafile_name, 'img': transformed_image})

    output_json, a_img, final_counts = run_model(d, mask=True)  # dir = directory to file containing images
    return output_json, a_img, final_counts

# def get_prediction(img):

#     preds = [mock_pred() for i in range(3)]
#     return preds

# def mock_pred():
#     return {
#         'predicted_class': random.randint(1,4),
#         'confidence': random.random(),
#         'bounding_box': [random.random(), random.random(), random.random(), random.random()]
#     }


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