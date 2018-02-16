import numpy as np
import pandas as pd
import seaborn as sns

import keras
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, Embedding, Concatenate, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D
from keras.models import Model


from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os, re, csv, math, codecs

sns.set_style("whitegrid")
np.random.seed(0)


path = '/Users/Frankie/PycharmProjects/Kaggle/Toxic_Comment_Classification/'
TRAIN_DATA_FILE = pd.read_csv(path+'train.csv')
TEST_DATA_FILE = pd.read_csv(path+'test.csv')

MAX_NB_WORDS = 40000
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])


#load embeddings
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open(path+'crawl-300d-2M.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


#load data
train_df = TRAIN_DATA_FILE
test_df = TEST_DATA_FILE
test_df = test_df.fillna('_NA_')

label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train_df[label_names].values

#visualize word distribution
train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)

raw_docs_train = train_df['comment_text'].tolist()
raw_docs_test = test_df['comment_text'].tolist()
num_classes = len(label_names)

print("pre-processing train data...")
processed_docs_train = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))
#end for

processed_docs_test = []
for doc in tqdm(raw_docs_test):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_test.append(" ".join(filtered))
#end for

print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))

#pad sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

#training params
batch_size = 128
num_epochs = 2
embed_dim = 300

#embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#CNN architecture
print("training LSTM ...")
inp = Input(shape=(max_seq_len,))
x = Embedding(MAX_NB_WORDS, embed_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.2, recurrent_dropout=0.4))(x)
x1 = GlobalMaxPool1D()(x)
x2 = GlobalAveragePooling1D()(x)
x = Concatenate(axis = 1)([x1,x2])
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(word_seq_train, y_train, batch_size=128, epochs=2, validation_split=0.1);

y_test = model.predict([word_seq_test],batch_size=1024,verbose=1)

sample_submission = pd.read_csv(path + 'sample_submission.csv')
sample_submission[label_names] = y_test
sample_submission.to_csv('submission_LSTM_FFNN_02.csv', index=False)