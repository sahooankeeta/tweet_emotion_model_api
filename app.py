import requests
from flask import Flask,json
import os.path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
# import tweet_model
app = Flask(__name__)



model=load_model('tweet_emotion_model.h5')
tokenizer=Tokenizer(num_words=10000,oov_token='<UNK>')
maxlen=60
class_to_index={'fear': 0, 'love': 1, 'sadness': 2, 'anger': 3, 'surprise': 4, 'joy': 5}
index_to_class=dict((v,k) for k,v in class_to_index.items())
filename='tweet_emotion_model.h5'

names_to_ids=lambda labels: np.array([class_to_index.get(x) for x in labels])

def get_sequence(tokenizer,tweets):
  sequences=tokenizer.texts_to_sequences(tweets)
  padded=pad_sequences(sequences,truncating='post',padding='post',maxlen=maxlen)
  return padded

@app.route("/")
def home():
   return "welcome"

@app.route("/predict",methods=['POST'])
def submit():
  input_data =requests.response()
  input_dictionary = input_data.json()
  text=input_dictionary['tweet']
  t=get_sequence(tokenizer,[text])
  p=model.predict(np.expand_dims(t[0],axis=0))[0]
  pred_class=index_to_class[np.argmax(p).astype('uint8')]
  return pred_class

if __name__=='main':
  app.run()