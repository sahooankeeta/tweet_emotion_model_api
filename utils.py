import tensorflow as tf
import numpy as np
import nlp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset=nlp.load_dataset('emotion')

train=dataset['train']
test=dataset['test']
val=dataset['validation']

def get_tweet(data):
  tweets=[x['text'] for x in data]
  labels=[x['label'] for x in data]
  return tweets,labels

tweets,labels=get_tweet(train)
tokenizer=Tokenizer(num_words=10000,oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)
maxlen=60

def get_sequence(tokenizer,tweets):
  sequences=tokenizer.texts_to_sequences(tweets)
  padded=pad_sequences(sequences,truncating='post',padding='post',maxlen=maxlen)
  return padded

padded_train_seq=get_sequence(tokenizer,tweets)

tokenizer=Tokenizer(num_words=10000,oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)
maxlen=60

class_to_index={'fear': 0, 'love': 1, 'sadness': 2, 'anger': 3, 'surprise': 4, 'joy': 5}
index_to_class=dict((v,k) for k,v in class_to_index.items())

names_to_ids=lambda labels: np.array([class_to_index.get(x) for x in labels])

train_labels=names_to_ids(labels)