from flask import Flask, request, redirect, url_for, flash, jsonify
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


app = Flask(__name__)

@app.before_first_request
def load_models():
    #Load TSV file for image_ids
    global tfidf, classifier
    print('Models Initiated')
    tfidf = pickle.load(open('./data/finalized_model_tfidf.sav', 'rb'))
    classifier = pickle.load(open('./data/finalized_model_XGB.sav', 'rb'))
    print('Models Loaded')


    

@app.route('/query-by-text/', methods=['POST'])
def make():


    data = request.get_json()
    print('Data Loaded, Type:', type(data))

    test = tfidf.transform(pd.Series(data))
    print(test)
    predict = classifier.predict(test)
    print('Type:',type(predict))
    print('This is the response:', predict[0])

    if predict[0] == 0:
        print('Negetive')
        a = 'Negetive'
    else:
        print('Positive')
        a = 'Positive'


    
    # Return index of the closest image.

    return jsonify(a)



if __name__ == '__main__':
    # The port number MUST match the one specified in `routes.js`.
    app.run(debug=True, host='0.0.0.0', port=50001)
