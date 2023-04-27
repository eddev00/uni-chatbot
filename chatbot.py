# Goal of this chatbot is to respond to FAQ of professors and students of FSA ait melloul


# Importing the Libraries
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
Ddata['patterns'] = Ddata['patterns'].apply(
    lambda word: [ltrs.lower() for ltrs in word if ltrs not in string.punctuation])
Ddata['patterns'] = Ddata['patterns'].apply(lambda word: ''.join(word))


Ddata
# Removing punctuations
