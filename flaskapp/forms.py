from flask_wtf import FlaskForm
from wtforms import BooleanField, SubmitField
from flask_wtf.file import FileField, FileAllowed, FileRequired

class UploadImageForm(FlaskForm):
    data_file = FileField('Upload Image File', render_kw={'multiple': True}, validators=[FileRequired(), FileAllowed(['png', 'jpeg', 'jpg'], 'Image Files only!')])
    mask = BooleanField('Mask', default=True)
    submit = SubmitField('Upload')

class PreviewImageForm(FlaskForm):
    data_file_preview = FileField('Upload Image File', validators=[FileRequired(), FileAllowed(['png', 'jpeg', 'jpg'], 'Image Files only!')])
    submit_preview = SubmitField('Preview')