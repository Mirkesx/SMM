#!/home/mc/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-

import cgitb
cgitb.enable()

#Created on Sun Feb 23 19:46:33 2020

#@author: Marco Cavalli X81000445

#LIBRERIE USATE

import pandas as pd
import re
from IPython.display import display, HTML
import textacy.preprocessing as txt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from stop_words import get_stop_words
import warnings
warnings.simplefilter('ignore')

pd.set_option('display.max_rows', 100)
pd.set_option('min_rows', 100)
pd.set_option('display.max_columns', 12)
pd.set_option('display.precision', 3)
pd.set_option('expand_frame_repr', False)

STOP_WORDS = get_stop_words('italian')
STOP_WORDS_H = 0.7
STOP_WORDS_L = 0.0
N_GRAMS = (1,3)

#Applica operazioni di preprocessing sul dataset
def clean(dataset):
    tweet_text = dataset['text'].values
    clean_text = cleanTweet(tweet_text)
    clean_df = pd.DataFrame(clean_text, columns=['x'])
    clean_df['label'] = dataset['handle'].values
    
    return clean_df

#Divide il dataset in TrS, TeS ed eventualmente VaS
def splitDataset(dataset,percent=0.25,knn=False,knn_percent=0.33):

    train_set, test_set, y_train_set, y_test_set  = train_test_split(dataset['x'].tolist(),dataset['label'].tolist(), test_size=percent, random_state=1234)

    return train_set, y_train_set, test_set, y_test_set

#Regressore Logistico
def LRegr(dataset, num_label, features=20000, tdf=True):
    
    train_set, y_train_set, test_set, y_test_set = splitDataset(dataset)
    
    count_vect = CountVectorizer(ngram_range=N_GRAMS, max_features=features, max_df=STOP_WORDS_H, min_df=STOP_WORDS_L, strip_accents='unicode', stop_words = STOP_WORDS)
    tfidf = TfidfTransformer(use_idf=tdf)

    
    if num_label > 2:
        log = LogisticRegression(solver='liblinear', multi_class='ovr')
    else:
        log = LogisticRegression(solver='liblinear')
        
    x_train_counts = count_vect.fit_transform(train_set)
    #x_train = tfidf.fit_transform(x_train_counts)
    
    x_test_counts = count_vect.transform(test_set)
    #x_test = tfidf.transform(x_test_counts)

    log.fit(x_train_counts, y_train_set)
    #log.fit(x_train, y_train_set)

    y_train_preds = log.predict(x_train_counts)
    y_test_preds = log.predict(x_test_counts)

    #print("F1 training scores: {:0.2f}".format(f1_score(y_train_set,y_train_preds,average='weighted')))
    #print("F1 test scores: {:0.2f}".format(f1_score(y_test_set,y_test_preds,average='weighted')))
    
    return log, tfidf, count_vect

    
#Stampa la lista dei dataframe salvati in memoria
def printDF(loc):
    import os
    directory = "../WEB-DATA/",loc
    fileList = os.listdir(directory)
    try:
        fileList =  [re.search('[A-Za-z]*[^.csv]', x).group(0) if x != '.csv' else '' for x in fileList]
    except Exception as e:
        print(e)
    print(fileList)
    
#Restituisce la lista dei dataframe salvati in memoria
def getDF(loc):
    import os
    directory = "../WEB-DATA/"+loc
    fileList = os.listdir(directory)
    try:
        fileList =  [re.search('[A-Z_a-z09]*[^.csv]', x).group(0) if x != '.csv' else '' for x in fileList]
    except Exception as e:
        print(e)
    return(fileList)

#Aggiunge un utente (se esiste) nel Dataset
def addUser(loc,labels):
    directory = "WEB-DATA/"+loc
    dataset = pd.DataFrame()
    for x in labels:
        user_df = pd.read_csv('../{}/{}.csv'.format(directory,x))
        dataset = pd.concat([dataset, user_df], axis=0, sort=True)
        #print("Aggiunto l'utente {}!".format(x))
    dataset = dataset.sort_values(by='retweet_count')
    dataset = dataset.drop(columns=['Unnamed: 0'])
    dataset = dataset.reset_index(drop=True)
    return dataset

#Rimuove un utente (se presente) dal Dataset
def dropUser(username,dataset,users_in_dt):
    print("Rimosso l'utente {}!".format(username))
    dt = dataset[dataset['handle'] != username]
    dt = dt.reset_index(drop=True)
    return dt
        

    
#ripuliamo una lista di tweet usando textacy
def cleanTweet(raw_data):
    data = [txt.replace_urls(x,"") for x in raw_data]
    data = [txt.replace_emails(x,"") for x in data]
    data = [txt.replace_emojis(x,"") for x in data]
    data = [txt.replace_user_handles(x,"") for x in data]
    data = [txt.replace_phone_numbers(x,"") for x in data]
    data = [txt.replace_numbers(x,"") for x in data]
    data = [txt.replace_currency_symbols(x,"") for x in data]
    data = [txt.replace_hashtags(x,"") for x in data]
    return data


#MAIN

import sys

message = sys.argv[1]#'Lasciatemi cantare con la chitarra in mano lasciatemi cantare sono un italiano Buongiorno Italia gli spaghetti al dente e un partigiano come Presidente autoradio sempre nella mano destra'
type_mes = sys.argv[2]#'POLITICO'

dataset = pd.DataFrame()
labels = getDF(type_mes)

dataset = addUser(type_mes, labels)

clas, tfd, vect = LRegr(clean(dataset),len(labels))
    
mess_vect = vect.transform([message])
#mess_transformed = tfd.transform(mess_vect)

Probas_x = pd.DataFrame(clas.predict_proba(mess_vect), columns = clas.classes_)
result = pd.DataFrame([message], columns=['Messaggio'])
result[clas.classes_] = Probas_x
df = pd.DataFrame()
df['Max'] = result[clas.classes_.tolist()].idxmax(axis=1)

output = []

import eli5
s = eli5.show_weights(clas, vec=vect, target_names=clas.classes_).data
s = s.replace("\n","")

output.append(df.loc[0].tolist())
output.append(result[df.loc[0].tolist()[0]].values)
output.append([s])

s = eli5.show_prediction(clas, message, vec=vect, target_names=clas.classes_).data
s = s.replace("\n","")

output.append([s])

print(output)