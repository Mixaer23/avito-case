import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import string
import joblib
import nltk
from nltk import word_tokenize
import re

def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation 
                    else ' ' for ch in text])

def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)

def start(df_res):  
   
    nltk.data.path = ['/app/lib/']
    
    prep_text = [remove_multiple_spaces(remove_punctuation(text.lower())) 
                 for text in tqdm(df_res['description'])]


    df_res['text_prep'] = prep_text
    
    russian_stopwords = []

    with open('/app/lib/stopwords.txt', encoding='utf-8') as f:
        for line in f:
            russian_stopwords.append(line.strip('\n'))
    

    sw_texts_list = []
    for text in tqdm(df_res['text_prep']):
        tokens = word_tokenize(text)    
        tokens = [token for token in tokens 
                  if token not in russian_stopwords and token != ' ']
        text = " ".join(tokens)
        sw_texts_list.append(text)

    df_res['text_sw'] = sw_texts_list

    X_test = df_res['text_sw']

    pipeline_filename = "/app/lib/my_pipeline.pkl"

    logreg = joblib.load(pipeline_filename)

    np.set_printoptions(suppress=True)
    
    lr_probs = logreg.predict_proba(X_test)
    lr_probs = lr_probs[:, 0]

    df = pd.DataFrame(np.around(lr_probs, decimals=2))
    
    df.columns = ['prediction']
    # df.insert(0, 'index', df.index)
    print(df)
    
    return df

    # df_res.proba.astype(np.float16)

    # X_test.sort_values('index')

    # df = pd.read_csv('train.csv')

    # df = df[:100000]

    # probability = logreg.predict_proba(df.description)

    # df.loc[:,'prediction'] = np.round(probability[:,0], decimals=2)

    # df.loc[:,'label'] = 'good'
    # for i in tqdm(range(len(df.is_bad))):
    #     if df.is_bad[i] == 1:
    #         df.loc[:,'label'][i] = 'bad'

    # df[['description','label','prediction']]

    # df[['prediction']].to_excel('./prediction.xlsx')


