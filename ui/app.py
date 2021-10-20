from flask import Flask, render_template, flash, redirect, url_for, session, request, flash
from .forms import UploadImageForm
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
import pandas as pd
import sys

from flask_bootstrap import Bootstrap
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pokemon'
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadImageForm()
    if form.validate_on_submit():
        f = form.data_file.data
        print(f)
        filename = secure_filename(f.filename)
        print(filename)
        #flash('Upload success! Please wait while we run our model.', 'alert-success')
        return render_template('result_download.html')
    return render_template('home.html', form=form)

@app.route('/result_download', methods=['GET', 'POST'])
def result_download():
    return render_template('result_download.html')