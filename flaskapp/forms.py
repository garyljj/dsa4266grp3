from flask_wtf import FlaskForm
from wtforms import BooleanField, SubmitField
from flask_wtf.file import FileField, FileAllowed, FileRequired

class UploadImageForm(FlaskForm):
    data_file = FileField('Upload Image File', render_kw={'multiple': True}, validators=[FileRequired(), FileAllowed(['jpg'], 'Image Files only!')])
    mask = BooleanField('Mask (Tick to enable masking for prediction)', default=True)
    fast = BooleanField('Fast Prediction (Tick to enable faster weights. Smaller model which is up to 4x faster!)', default=True)
    submit = SubmitField('Upload')

class PreviewImageForm(FlaskForm):
    data_file_preview = FileField('Upload Image File', validators=[FileRequired(), FileAllowed(['jpg'], 'Image Files only!')])
    submit_preview = SubmitField('Preview')