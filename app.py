
import requests
from flask import Flask
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)


model=pickle.load(open('tweet_emotion_model','rb'))
tokenizer=Tokenizer(num_words=10000,oov_token='<UNK>')
maxlen=60
class_to_index={'fear': 0, 'love': 1, 'sadness': 2, 'anger': 3, 'surprise': 4, 'joy': 5}
index_to_class=dict((v,k) for k,v in class_to_index.items())

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
  r=requests.json()
  text=r['tweet']
  t=get_sequence(tokenizer,[text])
  p=model.predict(np.expand_dims(t[0],axis=0))[0]
  pred_class=index_to_class[np.argmax(p).astype('uint8')]
  return pred_class

if __name__=='main':
  app.run()