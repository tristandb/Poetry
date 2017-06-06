"""
Version that uses words.
"""

from __future__ import print_function

from keras.layers import Embedding
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from progress_bar import ProgressBar

import numpy as np
import random
import sys

# Constants
POEMEND = 'poemend'
POEMSTART = 'poemstart'
LINEEND = 'lineend'
UNKNOWN_WORD = 'unknownword'

# Add 'UNKNOWN_WORD' to the results
SKIP_UNKNOWN_WORD = False
LOWER_WORDS = True

# Minimum amount that a word has to occur before it is used in the vocabulary
min_word_count = 5
# Maximum length of a training sentence.
max_len = 10
# Time steps in a training sentence
time_step = 4

# Open the sonnet and prepare it by replacing characters.
text_input = open('../sonnets/misc/misc.txt').read()
text_input = text_input.replace("\n", " \n ")
text_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

# Split input into multiple sonnets
texts = text_input.split(" \n  \n ")

splitted_texts = []
print("Preparing sonnets")
progressbar = ProgressBar(len(texts))
for text in texts:
    splitted_text = POEMSTART + " "
    text = text_to_word_sequence(text, filters=text_filter, lower=LOWER_WORDS, split=" ")
    for word in text:
        if word == "\n":
            word = LINEEND
            splitted_text += " " + word
            continue
        splitted_text += " " + word
    splitted_text += " " + POEMEND
    splitted_texts.append(splitted_text)
    progressbar.count()
print("")

# Create an initial tokenizer
text_tokenizer = Tokenizer(filters=text_filter, lower=LOWER_WORDS, split=" ", char_level=False)
text_tokenizer.fit_on_texts(splitted_texts)

# Generate a list of words that occur more than n times
# Generate a list of words that occur less than n times
less_occurring_words = []
more_occurring_words = []
print("Determining more occuring words")
progressbar = ProgressBar(len(text_tokenizer.word_counts.items()))
for word in text_tokenizer.word_counts.items():
    progressbar.count()
    if word[1] > min_word_count:
        if word != LINEEND and word != POEMSTART and word != POEMEND:
            more_occurring_words.append(word[0])
    else:
        less_occurring_words.append(word[0])
print("")

# Add default words
more_occurring_words.append(UNKNOWN_WORD)

print(more_occurring_words)

# Replace words that occur less than n times with UNKNOWN_WORD
print("Removing less occurring words")
progressbar = ProgressBar(len(texts))
splitted_texts = []
for i, text in enumerate(texts):
    splitted_text = [POEMSTART]
    text = text_to_word_sequence(text, filters=text_filter, lower=LOWER_WORDS, split=" ")
    for word in text:
        if word == "\n" or word == LINEEND:
            word = LINEEND
            splitted_text.append(word)
            continue
        if word in less_occurring_words:
            word = UNKNOWN_WORD
        splitted_text.append(word)
    splitted_text.append(POEMEND)
    progressbar.count()
    splitted_texts.append(splitted_text)
print("")

# Generate time-steps
progressbar = ProgressBar(len(splitted_texts))
generated_timesteps = []
next_words = []
for splitted_text in splitted_texts:
    for i in range(0, len(splitted_text) - max_len, time_step):
        generated_timesteps.append(splitted_text[i: i + max_len])
        next_words.append(splitted_text[i + max_len])
    progressbar.count()
print("")

words_set = set(more_occurring_words)
word_indices = dict((c, i) for i, c in enumerate(words_set))
indices_word = dict((i, c) for i, c in enumerate(words_set))
print(len(more_occurring_words))
print(more_occurring_words)
print(len(words_set))
print("Length", len(word_indices))
# Vectorization
X = np.zeros((len(generated_timesteps), max_len, len(more_occurring_words)), dtype=np.bool)
y = np.zeros((len(generated_timesteps), len(more_occurring_words)), dtype=np.bool)

for i, generated_timestep in enumerate(generated_timesteps):
    for t, word in enumerate(generated_timestep):
        if word != UNKNOWN_WORD:
            X[i, t, word_indices[word]] = 1
    if next_words[i] != UNKNOWN_WORD:
        y[i, word_indices[next_words[i]]] = 1

print('Building model')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(word_indices))))
# model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
# model.add(Dropout(0.2))
model.add(Dense(len(word_indices)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # https://github.com/llSourcell/How-to-Generate-Music-Demo/issues/4
    a = np.log(a) / temperature
    dist = np.exp(a) / np.sum(np.exp(a))
    choices = range(len(a))
    return np.random.choice(choices, p=dist)


# Train the model, output generated text after each iteration.
for iteration in range(1, 300):
    print()
    print('-' * 50)
    print('Iteration ', iteration)
    model.fit(X, y, batch_size=128, epochs=1)

    # Dump the model to disk
    json_string = model.to_json()
    open("arch.json", "w").write(json_string)

    model.save_weights("weights.h5", overwrite=True)

    start_index = random.randint(0, len(splitted_texts) - 1)

    for diversity in [0.4, 0.6, 0.8, 1.0, 1.2]:
        print()
        print('----- diversity: ', diversity)

        generated = ''
        sentence = [POEMSTART]

        print('----- Generating with seed: "' + " ".join(sentence).replace(LINEEND, "\n") + '"')
        sys.stdout.write(generated)

        for i in range(1000):
            x = np.zeros((1, max_len, len(more_occurring_words)))

            for t, word in enumerate(sentence):
                if word != UNKNOWN_WORD:
                    x[0, t, word_indices[word]] = 1.

            predicates = model.predict(x, verbose=0)[0]
            next_index = sample(predicates, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            if next_word == LINEEND:
                next_word = '\n'
            if next_word == POEMEND:
                break

            generated += " " + next_word

            sys.stdout.write(" " + next_word)
            sys.stdout.flush()
        print()
