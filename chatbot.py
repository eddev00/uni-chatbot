# Goal of this chatbot is to respond to FAQ of professors and students of FSA ait melloul


# Importing the Libraries
import pickle
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

# exporting the label encoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

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


while True:
    texts_p = []
    prediction_input = input("You: ")

    # Removing punctuations and converting to lowercase
    prediction_input = [
        ltrs.lower() for ltrs in prediction_input if ltrs not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    # Tokenizing the input
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)

    prediction_input = pad_sequences(
        [prediction_input], input_shape)

    # predicting the output
    output = model.predict(prediction_input)
    output = np.argmax(output)

    # Getting the tag
    response_tag = le.inverse_transform([output])[0]

    import chatbot

    responses = chatbot.responses
    import random
    print("Bot: ", random.choice(responses[response_tag]))
    if response_tag == 'goodbye':
        break


# Saving the Model
# h5py needs to be installed

try:
    model.save('chatbot.h5')
    print("Model saved successfully")
except Exception as error:
    print("Error saving model", error)
