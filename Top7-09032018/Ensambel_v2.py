import numpy as np, pandas as pd


path = '/Users/Frankie/PycharmProjects/Kaggle/Toxic_Comment_Classification/PreSubmission/'
f_GRU_fasttext = pd.read_csv(path+'submission_GRU_FF.csv')
f_baseline2 = pd.read_csv(path+'submission_baseline_2.csv')
f_BiGRUCNNGlove = pd.read_csv(path+'BiGRUCNNGlove.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = f_baseline2.copy()
p_res[label_cols] = (2*f_GRU_fasttext[label_cols] +
                     f_BiGRUCNNGlove[label_cols] +
                     3*f_baseline2[label_cols])/6

p_res.to_csv('submission_ensamble_x.csv', index=False)