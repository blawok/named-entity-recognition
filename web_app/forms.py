from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
from wtforms.validators import DataRequired

class TextForm(FlaskForm):
    """
    class that inherits from FlaskForm
    used for getting values from user
    """
    sentence_for_ner = TextField("Input sentence: ",
                                 validators=[DataRequired()])

    submit = SubmitField('Submit')
