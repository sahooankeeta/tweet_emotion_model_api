import tensorflow as tf
import numpy as np
import nlp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os.path
import utils
# dataset=nlp.load_dataset('emotion')

# train=dataset['train']
# test=dataset['test']
# val=dataset['validation']

# def get_tweet(data):
#   tweets=[x['text'] for x in data]
#   labels=[x['label'] for x in data]
#   return tweets,labels

# tweets,labels=get_tweet(train)
# tokenizer=Tokenizer(num_words=10000,oov_token='<UNK>')
# tokenizer.fit_on_texts(tweets)
# maxlen=60

# def get_sequence(tokenizer,tweets):
#   sequences=tokenizer.texts_to_sequences(tweets)
#   padded=pad_sequences(sequences,truncating='post',padding='post',maxlen=maxlen)
#   return padded

# padded_train_seq=get_sequence(tokenizer,tweets)

# classes=set(labels)
# class_to_index=dict((c,i) for i,c in enumerate(classes))
# index_to_class=dict((v,k) for k,v in class_to_index.items())
# class_to_index={'fear': 0, 'love': 1, 'sadness': 2, 'anger': 3, 'surprise': 4, 'joy': 5}
# index_to_class=dict((v,k) for k,v in class_to_index.items())

# names_to_ids=lambda labels: np.array([class_to_index.get(x) for x in labels])

# train_labels=names_to_ids(labels)
model=tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000,16,input_length=utils.maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6,activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
val_tweets,val_labels=utils.get_tweet(utils.val)
val_seq=utils.get_sequence(utils.tokenizer,val_tweets)
val_labels=utils.names_to_ids(val_labels)
h=model.fit(
    utils.padded_train_seq,utils.train_labels,
    validation_data=(val_seq,val_labels),
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)
    ]
)
test_tweets,test_labels=utils.get_tweet(utils.test)
test_seq=utils.get_sequence(utils.tokenizer,test_tweets)
test_labels=utils.names_to_ids(test_labels)
_=model.evaluate(test_seq,test_labels)

model.save('tweet_emotion_model.h5')
text="great to see you"
t=utils.get_sequence(utils.tokenizer,[text])
print(t)
p=model.predict(np.expand_dims(t[0],axis=0))[0]
pred_class=utils.index_to_class[np.argmax(p).astype('uint8')]
print(pred_class)