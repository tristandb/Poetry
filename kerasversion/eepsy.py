"""
Version that implements syllables.
https://github.com/karpathy/char-rnn/blob/master/train.lua
https://github.com/larspars/word-rnn
https://github.com/hunkim/word-rnn-tensorflow
https://github.com/hunkim/wpoem
https://larseidnes.com/2015/10/13/auto-generating-clickbait-with-recurrent-neural-networks/
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

Other:

This drop of performance is unlikely due to the difficulty for character level model to capture longer short term memory, since also the Longer Short Term Memory (LSTM) recurrent networks work better with word-based input.
one of the fundamental differences between the word level and character level models is in the number of parameters the RNN has to access during the training and test. The smaller is the input and output layer of RNN, the larger needs to be the fully connected hidden layer, which makes the training of the model expensive.
http://arxiv.org/abs/1511.06303

Remove infrequent words. Having a huge vocabulary will make our model slow to train. Replace it by an UNKNOWN_TOKEN.
After generation, replace UNKNOWN_TOKEN by a random sample from the deleted token.

Replace \n by SENTENCE_END and capital by SENTENCE_START?

Add expected loss calculations. (http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano)

remove unknown_word from y labels
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

# Minimum amount that a word has to occur before it is used in the vocabulary
min_word_count = 10
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
progressbar = ProgressBar(len(texts))
for text in texts:
    splitted_text = "POEMSTART "
    text = text_to_word_sequence(text, filters=text_filter, lower=True, split=" ")
    for word in text:
        if word == "\n":
            word = "LINEEND"
            splitted_text += " " + word
            continue
        for syllable in hyphenate_word(word):
            splitted_text += " " + syllable
    splitted_text += " POEMEND"
    splitted_texts.append(splitted_text)
    progressbar.count()
print("")

# Create an initial tokenizer
text_tokenizer = Tokenizer(filters=text_filter, lower=True, split=" ", char_level=False)
text_tokenizer.fit_on_texts(splitted_texts)

# Generate a list of words that occur more than n times
# Generate a list of words that occur less than n times
less_occurring_words = []
more_occurring_words = []
progressbar = ProgressBar(len(text_tokenizer.word_counts.items()))
for word in text_tokenizer.word_counts.items():
    progressbar.count()
    if word[1] > min_word_count:
        if word != 'lineend' and word != 'poemstart' and word != 'poemend':
            more_occurring_words.append(word[0])
    else:
        less_occurring_words.append(word[0])
print("")

# Add default words
more_occurring_words.append("UNKNOWN_WORD")
more_occurring_words.append("LINEEND")
more_occurring_words.append("POEMEND")
more_occurring_words.append("POEMSTART")

# Replace words that occur less than n times with UNKNOWN_WORD
progressbar = ProgressBar(len(texts))
splitted_texts = []
for i, text in enumerate(texts):
    splitted_text = []
    splitted_text.append("POEMSTART")
    text = text_to_word_sequence(text, filters=text_filter, lower=True, split=" ")
    for word in text:
        if word == "\n" or word == "lineend":
            word = "LINEEND"
            splitted_text.append(word)
            continue
        for syllable in hyphenate_word(word):
            if syllable in less_occurring_words:
                syllable = "UNKNOWN_WORD"
                splitted_text.append(syllable)
                continue
            splitted_text.append(syllable)
    splitted_text.append("POEMEND")
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

# Vectorization
X = np.zeros((len(generated_timesteps), max_len, len(more_occurring_words)), dtype=np.bool)
y = np.zeros((len(generated_timesteps), len(more_occurring_words)), dtype=np.bool)

for i, generated_timestep in enumerate(generated_timesteps):
    for t, word in enumerate(generated_timestep):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

print('Building model')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(word_indices))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(word_indices)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print(model.summary())

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
    open("arch_sy.json", "w").write(json_string)

    model.save_weights("weights_sy.h5", overwrite=True)

    start_index = random.randint(0, len(splitted_texts) - 1)

    for diversity in [0.8, 1.0, 1.2]:
        print()
        print('----- diversity: ', diversity)

        generated = ''
        sentence = ['POEMSTART']
        # sentence = splitted_texts[start_index][0: max_len]
        generated += " ".join(sentence)

        print('----- Generating with seed: "' + " ".join(sentence).replace("LINEEND", "\n") + '"')
        sys.stdout.write(generated)

        for i in range(1000):
            x = np.zeros((1, max_len, len(more_occurring_words)))

            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            if next_word == 'LINEEND':
                next_word = '\n'

            if next_word == 'POEMEND':
                break

            generated += " " + next_word

            sys.stdout.write(" " + next_word)
            sys.stdout.flush()
        print()
