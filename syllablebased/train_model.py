"""
Version that uses syllables.
"""

from __future__ import print_function

from keras.layers import Embedding
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from progress_bar import ProgressBar
from hyphenate import hyphenate_word

import numpy as np
import random
import sys

# Constants, used as delimiter
POEMEND = 'poemend'
POEMSTART = 'poemstart'
LINEEND = 'lineend'
UNKNOWN_WORD = 'unknownword'
SPACE_DELIMITER = 'spacedelimiter'


# Add 'UNKNOWN_WORD' to the results
SKIP_UNKNOWN_WORD = False
LOWER_WORDS = True

# Minimum amount that a syllable has to occur before it is used in the vocabulary
min_syllable_count = 5
# Maximum length of a training sentence.
max_len = 15
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
    splitted_text = POEMSTART
    text = text_to_word_sequence(text, filters=text_filter, lower=LOWER_WORDS, split=" ")
    for word in text:
        if word == "\n":
            splitted_text += " " + LINEEND
            continue

        # Word should not be hyphenated if it is a delimiter.
        if word not in [POEMEND, POEMSTART, LINEEND, UNKNOWN_WORD]:
            for i, syllable in enumerate(hyphenate_word(word)):
                # Specify that the syllable is at the beginning of a word.
                if i == 0:
                    splitted_text += " " + SPACE_DELIMITER + " " + syllable
                else:
                    splitted_text += " " + syllable
        else:
            splitted_text += " " + word

    splitted_text += " " + POEMEND
    splitted_texts.append(splitted_text)
    progressbar.count()
print("")

print(splitted_texts)

# Create an initial tokenizer
text_tokenizer = Tokenizer(filters=text_filter, lower=LOWER_WORDS, split=" ", char_level=False)
text_tokenizer.fit_on_texts(splitted_texts)

# Generate a list of syllables that occur more than n times
# Generate a list of syllables that occur less than n times
less_occurring_syllables = []
more_occurring_syllables = []
print("Determining more occurring syllables")
progressbar = ProgressBar(len(text_tokenizer.word_counts.items()))
for syllable in text_tokenizer.word_counts.items():
    progressbar.count()
    if syllable[1] > min_syllable_count:
        more_occurring_syllables.append(syllable[0])
    else:
        less_occurring_syllables.append(syllable[0])
print("")

# Add default words
more_occurring_syllables.append(UNKNOWN_WORD)

# Replace syllables that occur less than n times with UNKNOWN_WORD
print("Removing less occurring syllables")
progressbar = ProgressBar(len(texts))
splitted_texts = []
for i, text in enumerate(texts):
    splitted_text = [POEMSTART]
    text = text_to_word_sequence(text, filters=text_filter, lower=LOWER_WORDS, split=" ")
    for word in text:
        if word == "\n":
            splitted_text.append(LINEEND)
            continue

        splitted_text.append(SPACE_DELIMITER)
        for i, syllable in enumerate(hyphenate_word(word)):
            if syllable in less_occurring_syllables:
                syllable = UNKNOWN_WORD
            splitted_text.append(syllable)
    splitted_text.append(POEMEND)
    splitted_texts.append(splitted_text)
    progressbar.count()
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

words_set = set(more_occurring_syllables)
word_indices = dict((c, i) for i, c in enumerate(words_set))
indices_word = dict((i, c) for i, c in enumerate(words_set))
print(len(more_occurring_syllables))
print(more_occurring_syllables)
print(len(words_set))
print("Length", len(word_indices))

# Vectorization
X = np.zeros((len(generated_timesteps), max_len, len(more_occurring_syllables)), dtype=np.bool)
y = np.zeros((len(generated_timesteps), len(more_occurring_syllables)), dtype=np.bool)

for i, generated_timestep in enumerate(generated_timesteps):
    for t, word in enumerate(generated_timestep):
        if word != UNKNOWN_WORD:
            X[i, t, word_indices[word]] = 1
    if next_words[i] != UNKNOWN_WORD:
        y[i, word_indices[next_words[i]]] = 1

print('Building model')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(word_indices))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
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
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration ', iteration)
    model.fit(X, y, batch_size=128, epochs=1)

    # Dump the model to disk
    json_string = model.to_json()
    open("arch.json", "w").write(json_string)

    model.save_weights("weights.h5", overwrite=True)

    start_index = random.randint(0, len(splitted_texts) - 1)

    for diversity in [0.8, 1.0, 1.2]:
        print()
        print('----- diversity: ', diversity)

        generated = ''
        sentence = [POEMSTART]

        print('----- Generating with seed: "' + " ".join(sentence).replace(LINEEND, "\n") + '"')
        sys.stdout.write(generated)

        for i in range(1000):
            x = np.zeros((1, max_len, len(more_occurring_syllables)))

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
            if next_word == SPACE_DELIMITER:
                next_word = ' '

            generated += next_word

            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()
