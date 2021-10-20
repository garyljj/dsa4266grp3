from flask_wtf import FlaskForm, Form
from wtforms import *
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField, IntegerField, FileField, RadioField, FormField, FieldList, MultipleFileField, DecimalField
from wtforms.validators import ValidationError, DataRequired, EqualTo, Length
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
#from app.models import User
#from datetime import datetime,date,timedelta

class UploadImageForm(FlaskForm):
    data_file = FileField('Upload Image File', validators=[FileRequired(), FileAllowed(['png', 'jpeg', 'jpg'], 'Image Files only!')])
    submit = SubmitField('Upload')
