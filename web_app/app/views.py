from flask import (render_template, Flask, Markup, flash, redirect, request,
                   session, redirect, url_for)
from app import app

from forms import TextForm
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
from wtforms.validators import DataRequired

import numpy as np
import pandas as pd
import requests
import json
from flask import Flask, jsonify, request

tag2name = {14: 'B-art',
            16: 'B-eve',
            0: 'B-geo',
            13: 'B-gpe',
            12: 'B-nat',
            10: 'B-org',
            4: 'B-per',
            2: 'B-tim',
            5: 'I-art',
            7: 'I-eve',
            15: 'I-geo',
            8: 'I-gpe',
            11: 'I-nat',
            3: 'I-org',
            6: 'I-per',
            1: 'I-tim',
            17: 'X',
            9: 'O',
            18: '[CLS]',
            19: '[SEP]'}

@app.route('/', methods=['GET', 'POST'])
@app.route('/forms/', methods=['GET', 'POST'])
@app.route('/forms', methods=['GET', 'POST'])
def submit():
    form = TextForm()

    if form.validate_on_submit():
        session['sentence_for_ner'] = form.sentence_for_ner.data
        return redirect(url_for('predictions'))

    return render_template('index.html',
                           form=form)


@app.route("/predictions/", methods=['GET', 'POST'])
@app.route("/predictions", methods=['GET', 'POST'])
def predictions():

    sentence_for_ner = session['sentence_for_ner']
    resp = requests.post('https://9r58nr9e00.execute-api.eu-west-1.amazonaws.com/prod', data=sentence_for_ner)
    preds = json.loads(resp.content)

    pred_array = np.fromstring(str(preds)[1:-1], dtype=np.int, sep=', ')

    input_length = len(str(sentence_for_ner).split())
    df_preds = pd.DataFrame()
    df_preds['Tag'] = [tag2name[i] for i in pred_array[1:(input_length+1)].tolist()]
    df_preds['Word'] = sentence_for_ner.split()

    return render_template('predictions.html',
                           sentence_for_ner=sentence_for_ner,
                           preds=preds,
                           df_preds=df_preds.to_html(classes=["table table-bordered table-hover"]))
