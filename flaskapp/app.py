import os
import stat
import glob
import base64
import random
import json
import time
import numpy as np
import cv2
import shutil
from zipfile import ZipFile
from flask import Flask, render_template, redirect, url_for, session, request, flash, jsonify
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

        files = glob.glob(os.path.join(app.config['PREVIEW_FOLDER'], "*"))
        for f in files:
            os.chmod(f, stat.S_IRUSR | stat.S_IWUSR)
            os.remove(f)

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
        return render_template('home.html', hash="#masking", form1=form1, form2=form2, image_name=image_name, masked_image_name=masked_image_name, image_size1=image_size1, image_size2=image_size2)

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


            files = glob.glob(os.path.join(app.config['OUTPUT_FOLDER'], "*"))
            for f in files:
                os.chmod(f, stat.S_IRUSR | stat.S_IWUSR)
                os.remove(f)

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
            folder_in_zip = 'run_' + unique_num

            with ZipFile(os.path.join(app.config['OUTPUT_FOLDER'], zip_name), 'w') as z:
                for output, img, name in zip(output_json, a_img, all_names):
                    json_name = name + '.json'
                    image_name = name + '_annotated' + '.jpg'
                    with open(os.path.join(app.config['OUTPUT_FOLDER'], json_name), 'w') as f:
                        json.dump(output, f, indent=4)
                    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], image_name), img)
                    z.write(os.path.join(app.config['OUTPUT_FOLDER'], json_name), f'{folder_in_zip}/{json_name}')
                    z.write(os.path.join(app.config['OUTPUT_FOLDER'], image_name), f'{folder_in_zip}/{image_name}')
                
                final_counts.to_csv(os.path.join(app.config['OUTPUT_FOLDER'], 'counts.csv'), index=False)
                z.write(os.path.join(app.config['OUTPUT_FOLDER'], 'counts.csv'), f'{folder_in_zip}/counts.csv')

            ## Zipping everything in the output folder
            #shutil.make_archive(unique_num, 'zip', OUTPUT_FOLDER)

            return redirect(url_for('result_download'))
    return render_template('home.html', hash="", form1=form1, form2=form2)

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

    output_json, a_img, final_counts = run_model(d, mask=True)
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

    data = request.get_json()
    if data is None:
        return apiError('TypeError', 'Content type is not json')

    try:
        if not isinstance(data, dict):
            data = json.loads(request.get_json())
        filename = data['filename']
        imgbase64 = data['image_base64']
        mask = True if data.get('mask', True) else False

        img_binary = base64.b64decode(imgbase64)
        img = cv2.imdecode(np.frombuffer(img_binary, np.uint8), flags=1)

    except KeyError:
        return apiError('KeyError', "Missing either 'filename' or 'image_base64' field")
    except:
        return apiError('ParseError', 'Json data is invalid')

    if img is None:
        return apiError('ImageError', 'Invalid Base64 image provided')

    d = {
        'filename': filename,
        'img': img
    }
    
    print(f'mask={mask}')
    output_json, annotated_imgs, final_counts = run_model([d], mask=mask)

    data['prediction'] = output_json[0]['predictions']

    print(f'total time: {time.time() - start}')
    return jsonify(data)

@app.route('/testpage')
def test():
    return 'This is a testpage'

def apiError(errortype, message):
    return jsonify({
        'error': {
            'type': errortype,
            'message': message
        }
    }), 400