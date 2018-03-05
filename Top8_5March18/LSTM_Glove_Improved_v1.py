# -*- coding: utf-8 -*-
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Concatenate, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from attention_layers_keras import AttentionWeightedAverage
from keras.callbacks import EarlyStopping, ModelCheckpoint

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from keras import initializers, regularizers, constraints, optimizers, layers


path = '/Users/Frankie/PycharmProjects/Kaggle/Toxic_Comment_Classification/'
TRAIN_DATA_FILE = pd.read_csv(path+'train.csv')
TEST_DATA_FILE = pd.read_csv(path+'test.csv')
EMBEDDING_FILE= path+'glove.840B.300d.txt'

embed_size = 300 # how big is each word vector
max_features = 30000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 200 # max number of words in a comment to use


train = TRAIN_DATA_FILE
test = TEST_DATA_FILE


list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.2, recurrent_dropout=0.4))(x)
x = BatchNormalization()(x)
print(x.shape)
x1 = GlobalMaxPool1D()(x)
print(x1.shape)
x2 = GlobalAveragePooling1D()(x)
x = Concatenate(axis = 1)([x1,x2])
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_t, y, batch_size=128, epochs=2, validation_split=0.1);

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(path + 'sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission_LSTM_3.csv', index=False)