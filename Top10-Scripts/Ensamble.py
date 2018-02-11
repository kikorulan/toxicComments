import numpy as np, pandas as pd


path = '/Users/Frankie/PycharmProjects/Kaggle/Toxic_Comment_Classification/PreSubmission/'
f_lstm_glove = pd.read_csv(path+'submission_LSTM_Glove.csv')
f_ng = pd.read_csv(path+'submission_LG_Ngram.csv')
f_lstm_fasttext = pd.read_csv(path+'submission_LSTM_Fasttext.csv')
f_baseline = pd.read_csv(path+'submission_baseline.csv')
#f_nbsvm = pd.read_csv(path+'submission_NB-M.csv')


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = f_lstm_glove.copy()
p_res[label_cols] = (f_lstm_fasttext[label_cols] +
                     f_lstm_glove[label_cols] +
                     f_ng[label_cols] +
                     f_baseline[label_cols]) / 4


p_res.to_csv('submission_ensamble_new.csv', index=False)