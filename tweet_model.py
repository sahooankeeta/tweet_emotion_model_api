import tensorflow as tf
import numpy as np
import nlp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os.path

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

classes=set(labels)
class_to_index=dict((c,i) for i,c in enumerate(classes))
index_to_class=dict((v,k) for k,v in class_to_index.items())

names_to_ids=lambda labels: np.array([class_to_index.get(x) for x in labels])

train_labels=names_to_ids(labels)
model=tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000,16,input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6,activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
val_tweets,val_labels=get_tweet(val)
val_seq=get_sequence(tokenizer,val_tweets)
val_labels=names_to_ids(val_labels)
h=model.fit(
    padded_train_seq,train_labels,
    validation_data=(val_seq,val_labels),
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)
    ]
)
test_tweets,test_labels=get_tweet(test)
test_seq=get_sequence(tokenizer,test_tweets)
test_labels=names_to_ids(test_labels)
_=model.evaluate(test_seq,test_labels)

model.save('tweet_emotion_model.h5')