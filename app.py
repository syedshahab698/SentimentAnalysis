# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 04:40:44 2020

@author: Shahab
"""

#import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model_tfidf = pickle.load(open('tfidfVec.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.values()
    final_features = model_tfidf.transform(int_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    sentiment = {0:'Positive Comment',1:'Negative Comment'}

    return render_template('index.html', prediction_text='The Comment is most likely a {}'.format(sentiment[output]))


if __name__ == "__main__":
    app.run(debug=True)