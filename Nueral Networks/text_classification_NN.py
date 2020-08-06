# In real-world, normalizing by /255 is normally more difficult
# This program tells you what movie review was positive or negative

import tensorflow as tf
from tensorflow import keras
import numpy as np  # Have to use old numpy==1.16.1

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)  # Takes 10,000 most common words in DB

#print(train_data[0])  # Prints integer encoded words, so have to map these words

word_index = data.get_word_index()  # Gives dictionary for encoded int words, (outputs tuples)

word_index = {k: (v+3) for k, v in word_index.items()}  # + 3 for below special keys (basically not recognized keys)
word_index["<PAD>"] = 0  # Adds space so every review the same
word_index["<START>"] = 1  # Begining tag
word_index["<UNK>"] = 2  # Unknown character
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # Swaps keys and values so int points to word

# Normalizing data so model sees consistently formed data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)  # Max words for review, ignores past this or adds pads
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

#print(len(test_data), len(test_data))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])  # Default to ? for key errors

#print(decode_review(test_data[0]))  # Prints reviews
#print(len(test_data[0]), len(test_data[1]))  # We have two different lengths, bad for model, we need to define this for nuerons, 250 for both

# Model here (saved so don't need it anymore)
'''
model = keras.Sequential()  # Could put list of layers here - lists below are sequential
model.add(keras.layers.Embedding(88000, 16))  # 88K word vectors/#'s per word
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # loss: 0.2 output but close to 0 so 0

# Splitting data
x_val = train_data[:10000]  # 10k validation data (hyper tuning is best way to get good model)
x_train = train_data[10000:]

y_val = train_labels[:10000]  
y_train = train_labels[10000:]

fitmodel = model.fit(x_train, y_train, epochs=30, batch_size=512, validation_data=(x_val, y_val), verbose=1)  # Batch size = how many we will load in at onc, sometimes you have too many to put in memory

results = model.evaluate(test_data, test_labels)

print(results)  # loss, accuracy

model.save("model.h5")  # Saves model in keras/Tf
'''


def review_encode(s):
    encoded = [1]  # Start tag = 1
    for word in s:
        if word.lower() in word_index:  # Is word in vocab?
            encoded.append(word_index[word.lower()])  # Word.lower so capitalized words in vocab
        else:
            encoded.append(2)  # UNK tag if not in vocab
    return encoded


model = keras.models.load_model("model.h5")  # Could tweak models and train a bunch and save best one

with open("test_review.txt", encoding="utf-8") as f:  # utf-8 = standard text encoding, with so you don't close file after running
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace(";", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")  # strip removes\n
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)  # Max words for review, ignores past this or adds pads
        predict = model.predict(encode)
        print(line)  # The review
        print(encode)  # Encoded review
        print(predict[0])  # Model thinks its positive or negative (1 = pos 0 = neg)


'''
test_review = test_data[0]  # Takes first review
predict = model.predict([test_review])  # Tells model to predict (not in test data)
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
'''
