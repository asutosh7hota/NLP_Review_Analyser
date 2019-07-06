from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import json
import base64
import io
from PIL import Image
import numpy as np
import pickle
import pandas as pd
from keras import backend as K
import tensorflow as tf
import pandas as pd
from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
import random 
import time

'''
vector = tfidf.transform(pd.Series(data))
my_prediction = classifier.predict(vector)

if my_prediction[0] == 0:
    print('Negetive')
    a = 0
else:
    print('Positive')
    a = 1


'''

app = Flask(__name__)
Bootstrap(app)


@app.before_first_request
def load_models():
    #Load TSV file for image_ids
    global tfidf, classifier
    print('Models Initiated')
    tfidf = pickle.load(open('./data/finalized_model_tfidf.sav', 'rb'))
    classifier = pickle.load(open('./data/finalized_model_XGB.sav', 'rb'))
    print('Models Loaded')


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        #NLP Stuff
        vector = tfidf.transform(pd.Series(rawtext))
        my_prediction = classifier.predict(vector)

        if my_prediction[0] == 0:
            print('Negetive')
            b = 'Negetive'
            a = 0
        else:
            print('Positive')
            b = 'Positive'
            a = 1
    end = time.time()
    final_time = end -start
    return render_template('index.html',number_of_tokens = b)






if __name__ == '__main__':
	app.run(debug=True)