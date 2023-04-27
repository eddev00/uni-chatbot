# Goal of this chatbot is to respond to FAQ of professors and students of FSA ait melloul


# Importing the Libraries
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.sequence import pad_sequences
import string
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import nltk
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from keras.models import Model
import matplotlib.pyplot as plt

# Importing the Dataset
with open('intents.json') as file:
    data = json.load(file)

# Transforming the Dataset into a DataFrame
tags = []
patterns = []
responses = {}
for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['patterns']:
        patterns.append(line)
        tags.append(intent['tag'])

# Creating a DataFrame
Ddata = pd.DataFrame({'patterns': patterns, 'tags': tags})


# Preprocessing the Dataset
# Removing punctuations
Ddata['patterns'] = Ddata['patterns'].apply(
    lambda word: [ltrs.lower() for ltrs in word if ltrs not in string.punctuation])
Ddata['patterns'] = Ddata['patterns'].apply(lambda word: ''.join(word))


# Tokenizing the Dataset
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(Ddata['patterns'])
train = tokenizer.texts_to_sequences(Ddata['patterns'])

# Padding the Dataset
x_train = pad_sequences(train)


# encoding the output
le = LabelEncoder()
y_train = le.fit_transform(Ddata['tags'])

input_shape = x_train.shape[1]
print(input_shape)

# define vocabulary
vocab_size = len(tokenizer.word_index)
print("number of unique words: ", vocab_size)
output_length = le.classes_.shape[0]
print("output length: ", output_length)


# Building the Model
# Neural network model
i = Input(shape=(input_shape,))
x = Embedding(vocab_size+1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation='softmax')(x)
model = Model(i, x)


# Compiling the Model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Training the Model
train = model.fit(x_train, y_train, epochs=200)

# Plotting the Loss and Accuracy
plt.plot(train.history['loss'], label='loss')
plt.plot(train.history['accuracy'], label='accuracy')
plt.legend()
plt.show()


# Saving the Model
model.save('chatbot.h5')
