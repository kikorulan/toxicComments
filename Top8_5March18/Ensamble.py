import numpy as np, pandas as pd


path = '/Users/Frankie/PycharmProjects/Kaggle/Toxic_Comment_Classification/PreSubmission/'
f_lstm_glove = pd.read_csv(path+'submission_LSTM_Glove.csv')
f_ng = pd.read_csv(path+'submission_LG_Ngram.csv')
f_lstm_fasttext = pd.read_csv(path+'submission_LSTM_Fasttext.csv')
f_GRU_fasttext = pd.read_csv(path+'submission_GRU_FF.csv')
f_baseline1 = pd.read_csv(path+'submission_baseline_2.csv')
f_baseline2 = pd.read_csv(path+'submission_baseline_2.csv')
f_baseline3 = pd.read_csv(path+'submission_baseline_3.csv')
f_onemoreblend =  pd.read_csv(path+'submission_onemoreblend.csv')
#f_nbsvm = pd.read_csv(path+'submission_NB-M.csv')


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = f_lstm_glove.copy()
p_res[label_cols] = (2*f_GRU_fasttext[label_cols] +
                     f_lstm_glove[label_cols] +
                     3*f_baseline2[label_cols])/6


p_res.to_csv('submission_ensamble_N9.csv', index=False)