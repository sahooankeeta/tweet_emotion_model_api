import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app=FastAPI()

origins=["*"]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
class_to_index={'fear': 0, 'joy': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'love': 5}
index_to_class=dict((v,k) for k,v in class_to_index.items())

tweet_emotion_model=pickle.load(open('tweet_emotion_model.sav','rb'))

@app.post('/emotion_prediction')
def emotion_pred(input_parameters):
  input_data=input_parameters.json()
  # input_dict=json.loads(input_data)
  # tweet=input_dict['tweet']
  # p=tweet_emotion_model.predict(np.expand_dims(tweet,axis=0))[0]

  # pred_class=index_to_class[np.argmax(p).astype('uint8')]
  return input_data