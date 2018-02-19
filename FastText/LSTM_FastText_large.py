
#============================================
#    IMPORT MODULES
#============================================
import re
import numpy as np
import pandas as pd
from fastText import load_model
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
from keras.utils import multi_gpu_model


window_length = 50 # The amount of words we look at per example. Experiment with this.

#============================================
#    LOAD DATA
#============================================
def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|?«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s

print('\nLoading data')
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
train['comment_text'] = train['comment_text'].fillna('_empty_')
test['comment_text'] = test['comment_text'].fillna('_empty_')

#============================================
#    DEFINE CLASSES AND SPLIT WORDS
#============================================
classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

print('\nLoading FT model')
ft_model = load_model('../dataset/wiki.en.bin')
n_features = ft_model.get_dimension()

def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = normalize(text)
    words = text.split()
    window = words[-window_length:]
    
    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def df_to_data(df):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['comment_text'].values):
        x[i, :] = text_to_vector(comment)

    return x


#============================================
#    SPLIT DATA
#============================================
# Split the dataset:
split_index = round(len(train) * 0.9)
shuffled_train = train.sample(frac=1)
df_train = shuffled_train.iloc[:split_index]
df_val = shuffled_train.iloc[split_index:]

# Convert validation set to fixed array
x_val = df_to_data(df_val)
y_val = df_val[classes].values

def data_generator(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0 # Counter inside the current batch vector
    batch_x = None # The current batch's x data
    batch_y = None # The current batch's y data
    
    while True: # Loop forever
        df = df.sample(frac=1) # Shuffle df each epoch
        
        for i, row in df.iterrows():
            comment = row['comment_text']
            
            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                batch_y = np.zeros((batch_size, len(classes)), dtype='float32')
                
            batch_x[batch_i] = text_to_vector(comment)
            batch_y[batch_i] = row[classes].values
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_i = 0


#============================================
#    BUILD MODEL
#============================================

# CNN architecture
print("training LSTM ...")
input_shape = Input(shape=(window_length, n_features))
#x = (x)
#x1 = GlobalMaxPool1D()(x)
#x2 = GlobalAveragePooling1D()(x)
#x = Concatenate(axis = 1)([x1,x2])
#x = Dense(6, activation="sigmoid")(x)

model = Sequential()
#model.add(Bidirectional(LSTM(input_shape, return_sequences=True, dropout=0.2, recurrent_dropout=0.4)))
model.add(LSTM(input_shape=input_shape, units=n_features, return_sequences=True, dropout=0.2, recurrent_dropout=0.4))
model.add(Dense(6, activation="sigmoid"))
#model = Model(inputs=input_shape, outputs=x)

# Multi GPU model
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 128
training_steps_per_epoch = round(len(df_train) / batch_size)
training_generator = data_generator(df_train, batch_size)

# Ready to start training:
parallel_model.fit_generator(
    training_generator,
    steps_per_epoch=training_steps_per_epoch,
    batch_size=batch_size,
    validation_data=(x_val, y_val)
)


#============================================
#    PREDICTION AND SUBMISSION
#============================================
y_test = parallel_model.predict([word_seq_test],batch_size=1024,verbose=1)

sample_submission = pd.read_csv(path + 'sample_submission.csv')
sample_submission[label_names] = y_test
sample_submission.to_csv('submission_LSTM_FFNN_02.csv', index=False)





# import numpy as np
# import pandas as pd
# import seaborn as sns
# 
# import keras
# from keras import optimizers
# from keras import regularizers
# from keras.models import Sequential
# from keras.preprocessing import sequence
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.text import Tokenizer
# from keras.layers import Dense, Input, LSTM, Embedding, Concatenate, Dropout, Activation
# from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D
# from keras.models import Model
# 
# 
# from tqdm import tqdm
# from nltk.corpus import stopwords
# from nltk.tokenize import RegexpTokenizer
# import os, re, csv, math, codecs
# 
# sns.set_style("whitegrid")
# np.random.seed(0)
# 
# 
# path = '/Users/Frankie/PycharmProjects/Kaggle/Toxic_Comment_Classification/'
# TRAIN_DATA_FILE = pd.read_csv(path+'train.csv')
# TEST_DATA_FILE = pd.read_csv(path+'test.csv')
# 
# MAX_NB_WORDS = 40000
# tokenizer = RegexpTokenizer(r'\w+')
# stop_words = set(stopwords.words('english'))
# stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
# 
# 
# #load embeddings
# print('loading word embeddings...')
# embeddings_index = {}
# f = codecs.open(path+'crawl-300d-2M.vec', encoding='utf-8')
# for line in tqdm(f):
#     values = line.rstrip().rsplit(' ')
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
# 
# 
# #load data
# train_df = TRAIN_DATA_FILE
# test_df = TEST_DATA_FILE
# test_df = test_df.fillna('_NA_')
# 
# label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# y_train = train_df[label_names].values
# 
# #visualize word distribution
# train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))
# max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)
# 
# raw_docs_train = train_df['comment_text'].tolist()
# raw_docs_test = test_df['comment_text'].tolist()
# num_classes = len(label_names)
# 
# print("pre-processing train data...")
# processed_docs_train = []
# for doc in tqdm(raw_docs_train):
#     tokens = tokenizer.tokenize(doc)
#     filtered = [word for word in tokens if word not in stop_words]
#     processed_docs_train.append(" ".join(filtered))
# #end for
# 
# processed_docs_test = []
# for doc in tqdm(raw_docs_test):
#     tokens = tokenizer.tokenize(doc)
#     filtered = [word for word in tokens if word not in stop_words]
#     processed_docs_test.append(" ".join(filtered))
# #end for
# 
# print("tokenizing input data...")
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
# tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
# word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
# word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
# word_index = tokenizer.word_index
# print("dictionary size: ", len(word_index))
# 
# #pad sequences
# word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
# word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
# 
# #training params
# batch_size = 128
# num_epochs = 2
# embed_dim = 300
# 
# #embedding matrix
# print('preparing embedding matrix...')
# words_not_found = []
# nb_words = min(MAX_NB_WORDS, len(word_index))
# embedding_matrix = np.zeros((nb_words, embed_dim))
# for word, i in word_index.items():
#     if i >= nb_words:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if (embedding_vector is not None) and len(embedding_vector) > 0:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#     else:
#         words_not_found.append(word)
# print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
# 
# #CNN architecture
# print("training LSTM ...")
# inp = Input(shape=(max_seq_len,))
# x = Embedding(MAX_NB_WORDS, embed_dim, weights=[embedding_matrix])(inp)
# x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.2, recurrent_dropout=0.4))(x)
# x1 = GlobalMaxPool1D()(x)
# x2 = GlobalAveragePooling1D()(x)
# x = Concatenate(axis = 1)([x1,x2])
# x = Dense(6, activation="sigmoid")(x)
# model = Model(inputs=inp, outputs=x)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 
# model.fit(word_seq_train, y_train, batch_size=128, epochs=2, validation_split=0.1);
# 
# y_test = model.predict([word_seq_test],batch_size=1024,verbose=1)
# 
# sample_submission = pd.read_csv(path + 'sample_submission.csv')
# sample_submission[label_names] = y_test
# sample_submission.to_csv('submission_LSTM_FFNN_02.csv', index=False)
