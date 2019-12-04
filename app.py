from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pickle
from keras import backend as K
import tensorflow as tf
from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
import random 
import time
from sklearn.utils import shuffle
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


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
    global token, classifier,graph
    print('Models Initiated')
    token = pickle.load(open('tokenizer.pickle', 'rb'))
    classifier = pickle.load(open('SentimentAnalysisModel.pickle', 'rb'))
    global graph
    graph = tf.get_default_graph()
    print('Models Loaded')

@app.route('/music/1.aac')  
def send_file(filename):  
    return send_from_directory("/music", '1.aac')

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        #NLP Stuff
        
        with graph.as_default():
            y=token.texts_to_sequences([rawtext])
            y1 = pad_sequences(y, maxlen=128, padding='post')
            print('This is the padded sequence:',y1)
            print('Type:',type(y1))
            my_prediction=classifier.predict(y1)
            print('This is the prediction',my_prediction)
            if my_prediction[0][0] < my_prediction[0][1]:
                print('Positive')
                b = 'Positive'
                
            else:
                print('Negative')
                b = 'Negative'
                
    end = time.time()
    #final_time = end -start
    return render_template('index.html',number_of_tokens = b, neg= round(my_prediction[0][0],3), pos=round(my_prediction[0][1],3))






if __name__ == '__main__':
	app.run(debug=True)