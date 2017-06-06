from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard
import re
from hyphenate import hyphenate_word
from hurry.filesize import size

import numpy as np
import random
import sys
import json
import pickle

# read from file
with open('../sonnets/misc/misc_small.txt') as text:
    content = [line.rstrip() for line in text]

# Split up multidimensional array to a single dimensional array with newlines.
content = ["X" if word is not "\n" else "\n"for line in content for word in line.split(' ') + ['\n']]

# Remove all special characters from a single dimensional array, yield newlines.
content = [''.join(e for e in word.lower() if e.isalpha() or e == '\n') for word in content]


# split up syllables and add a space before every word.
content = [syllable for word in content for syllable in hyphenate_word(word)]

print('corpus length:', len(content))

# Create a character set.
chars = set()
[chars.add(word) for word in content]

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Remove characters that are not necessary
pickle.dump(char_indices, open("char_indic.json", "wb"))
pickle.dump(indices_char, open("indic_char.json", "wb"))

# Cut the text in semi-redundant sequences of maxlen characters
maxlen = 32
step = 2
sentences = []
next_chars = []


for i in range(0, len(content) - maxlen):
    if i > 0 and content[i-1] == '\n':
        sentences.append(content[i: i + maxlen])
        next_chars.append(content[i + maxlen])

print('nb sequences:', len(sentences))
print(size(len(sentences) * maxlen * 1))

print('Vectorization')

X = np.zeros((len(sentences), maxlen, 1), dtype=np.int)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    # TODO: Split up in syllables instead of
    for t, char in enumerate(sentence):
        X[i, t, 0] = char_indices[char]
    y[i, char_indices[next_chars[i]]] = 1

# Build the model 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, 1)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

tensorboard = TensorBoard(log_dir='./testlogs', histogram_freq=10000, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # https://github.com/llSourcell/How-to-Generate-Music-Demo/issues/4
    a = np.log(a) / temperature
    dist = np.exp(a) / np.sum(np.exp(a))
    choices = range(len(a))
    return np.random.choice(choices, p=dist)


# Train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, epochs=1, callbacks=[tensorboard])

    # Dump the model to disk
    json_string = model.to_json()
    open('arch.json', 'w').write(json_string)
    model.save_weights('testweights.h5', overwrite=True)

    start_index = random.randint(0, len(sentences) - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        initial_sentence = sentences[start_index]
        print(initial_sentence)
        joined_initial_sentence = ''.join(initial_sentence)
        print('----- Generating with seed: "' + joined_initial_sentence + '"')
        # sys.stdout.write(generated)

        for i in range(50):
            x = np.zeros((1, maxlen, 1))
            for t, char in enumerate(initial_sentence):
                x[0, t, 0] = char_indices[char]
            preds = model.predict(x, verbose=0)[0]

            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char

            length = len(initial_sentence) + 1
            initial_sentence = initial_sentence[length:] + [next_char]

            #sys.stdout.write(next_char)
            #sys.stdout.flush()
        print(''.join(generated))


