import pandas as pd
import numpy as np
import nltk
import pyarabic.arabrepr
from tashaphyne.stemming import ArabicLightStemmer

def file(file_pass):
    df = pd.read_csv(file_pass)
    return df
file('processed_data.csv')
df = file('processed_data.csv')
print(df)

def segmentation(df):
    df['Content'] = str(df['Content'])
    sent_token=nltk.data.load('tokenizers/punkt/english.pickle')
    for qi in df['Content']:
        df['Content'][qi]=sent_token.tokenize(qi)
    return df
df = segmentation(df)
print(df)

def Tokenization(df):
    df['Content'] = df['Content'].apply(nltk.word_tokenize)
    return df
df = Tokenization(df)
print(df)

def stopwords_Removing(df):
    stopw = pd.read_csv('arabic-stop-words.csv')
    gnewl=[]
    for i0 in df['Content']:
        newlist1=[]
        for word0 in i0:
            if word0 not in stopw:
                newlist1.append(word0)
        gnewl.append(newlist1)
    df['Content']=gnewl
    return df
df = stopwords_Removing(df)
print(df)

ArListem = ArabicLightStemmer()
def light_based_stemmer(df):
    df['Content'] = df['Content'].apply(lambda x: [ArListem.light_stem(y) for y in x])
    return df
df = light_based_stemmer(df)    
print(df)

def root_base_stemmer(df):
    nnewl=[]
    for i in df['Content']:
        newl=[]
        for word in i:
            stem = ArListem.light_stem(str(word))
            l =ArListem.get_root()
            newl.append(l)
        nnewl.append(newl)
    df['Content']=nnewl
    return df
df = root_base_stemmer(df)    
print(df)


