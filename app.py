import requests
from flask import Flask,json,request
import tensorflow as tf
import numpy as np
import nlp
import os.path
from tensorflow.keras.models import load_model
import utils

app = Flask(__name__)

model=load_model('tweet_emotion_model.h5')

filename='tweet_emotion_model.h5'

@app.route("/")
def home():
  return "welcome"

@app.route("/predict",methods=['POST'])
def submit():
  text=request.json['tweet']
  t=utils.get_sequence(utils.tokenizer,[text])
  p=model.predict(np.expand_dims(t[0],axis=0))[0]
  pred_class=utils.index_to_class[np.argmax(p).astype('uint8')]
  return pred_class

if __name__=='main':
  app.run()