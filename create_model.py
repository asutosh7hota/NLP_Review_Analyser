
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from keras.preprocessing.sequence import pad_sequences


from collections import Counter
import seaborn as sns
import string

import regex as re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.backend import eval
from keras.optimizers import Adam
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D,MaxPooling1D
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix
import seaborn as sns

def top_tfidf_feats(row, features, top_n=50):
   
    topn_ind = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ind]
    df = pd.DataFrame(top_feats,columns = ['feature', 'tv'])
    return df



clean_df=pd.read_csv("./clean_df.csv")
print('Data Shape:',clean_df.shape)
clean_df.isna().sum()
clean_df=clean_df.dropna()
clean_df = clean_df.loc[:, ~clean_df.columns.str.contains('^Unnamed')]
clean_df.shape
print('Cleaned Data:', clean_df.head())

x=clean_df["reviews"]
y=clean_df["labels"]

tv = TfidfVectorizer(stop_words ='english',ngram_range=(1,3),max_features=10000)
x_tv= tv.fit_transform(x)

#Create tfidf pickle and save
pickle.dump(tv, open('finalized_model_tfidf.sav', 'wb'))

features = tv.get_feature_names()
print('Features exist:',features[100:120])


top_tfidfs = top_tfidf_feats(x_tv[1000,:].toarray()[0],features, 50)

x_train_tv, x_test_tv, y_train_tv, y_test_tv = train_test_split(x_tv, y, test_size=0.2, random_state=0)

xg_clf = XGBClassifier()
xg_clf.fit(x_train_tv, y_train_tv)

y_pred=xg_clf.predict(x_test_tv)

pickle.dump(xg_clf, open('finalized_model_XGB.sav', 'wb'))

print('Model trained properly and saved')
