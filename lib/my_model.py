import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import string
import nltk
from nltk import word_tokenize
import re
import catboost
import emoji
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer 

def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])

def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])

def remove_multiple_spaces(text):
	return re.sub(r'\s+', ' ', text, flags=re.I)

def remove_emoji(text):
    return ''.join(char for char in text if char not in emoji.UNICODE_EMOJI)
def start(df_val):  

    stemmer = SnowballStemmer("russian")
   
    nltk.data.path = ['/app/lib/']
    
    prep_text_val = [remove_multiple_spaces(remove_emoji(remove_punctuation(text.lower()))) 
                     for text in tqdm(df_val['description'])]


    df_val['text_prep'] = prep_text_val
    
    russian_stopwords = []

    with open('/app/lib/stopwords.txt', encoding='utf-8') as f:
        for line in f:
            russian_stopwords.append(line.strip('\n'))
    

    stemmed_texts_list_val = []
    for text in tqdm(df_val['text_prep']):
        tokens = word_tokenize(text)    
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in russian_stopwords]
        text = " ".join(stemmed_tokens)
        stemmed_texts_list_val.append(text)

    df_val['text_stem'] = stemmed_texts_list_val

    model_filename = "/app/lib/pred_model"

    cat_model = catboost.CatBoostClassifier()

    model = cat_model.load_model(model_filename)
    
    lr_probs = model.predict_proba(df_val[['title', 'subcategory', 'category', 'region',
       'city', 'text_stem']])
    lr_probs = lr_probs[:, 1]

    df = pd.DataFrame(np.around(lr_probs, decimals=2))
    
    df.columns = ['prediction']

    print(df)
    
    return df
