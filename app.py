import numpy as np
from flask import Flask,request, url_for, redirect, jsonify
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = Flask(__name__)


class_to_index={'fear': 0, 'joy': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'love': 5}
index_to_class=dict((v,k) for k,v in class_to_index.items())

tweet_emotion_model=pickle.load(open('tweet_emotion_model.sav','rb'))

@app.route("/")
def home():
  return "hi there"

@app.route('/emotion_prediction',methods=['POST'])
def emotion_pred():
  print(request.form)
  input_data=request.form.tweet
  # input_dict=json.loads(input_data)
  # tweet=input_dict['tweet']
  # p=tweet_emotion_model.predict(np.expand_dims(tweet,axis=0))[0]

  # pred_class=index_to_class[np.argmax(p).astype('uint8')]
  return input_data

if __name__=='main':
  app.run()