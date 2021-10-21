import os
from flask import Flask, render_template, flash, redirect, url_for, session, request, flash
from .forms import UploadImageForm
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
import pandas as pd
import sys

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