# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:46:33 2020

@author: Marco Cavalli X81000445
"""

"""LIBRERIE USATE"""

import pandas as pd
import twitter
import re
import eli5
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import textacy.preprocessing as txt
from stop_words import get_stop_words 
import warnings
warnings.simplefilter('ignore')

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.precision', 3)
pd.set_option('expand_frame_repr', False)

eng_stop_words = get_stop_words('english')
ita_stop_words = get_stop_words('italian')
list_stop_words = eng_stop_words + ita_stop_words
STOP_WORDS_H = 0.7
STOP_WORDS_L = 0.0
N_GRAMS = (1,4)
api = None
dataset = None
users_in_dt = []
num_label = 0
classifier = tfidf = count_vect = None


"""MAIN"""


def main():
    api = None
    dataset = None
    users_in_dt = []
    num_label = 0
    classifier = tfidf = count_vect = None
    
    makeData()
    
    while True:    
        answer = input("\n\nMENU' PRINCIPALE\n0) Reinizializza le api\n1) Visualizza la lista dei DataFrame salvati in memoria\n2) Seleziona/Mina i tweet di un utente e salvali nel Dataset\n3) Rimuovi un utente dal Dataset\n4) Stampa la lista degli utenti presenti sul Dataset\n5) Visualizza il Dataset\n6) Scegli un classificatore\n7) Classifica un tweet\n8) Termina l'esecuzione\nScegli un'azione da compiere:")
        if answer == '0':
            api = setAPI()
        elif answer == '1':
            printDF()
        elif answer == '2':
            classifier = tfidf = count_vect = None
            user = input("Indica un username da aggiungere/minare: ")
            if user not in users_in_dt:
                err, api, user = mine_tweets(user,api,1500)
                if err == 1:
                    print("E' successo un imprevisto. Riprovare!")
                    continue
                dataset, err = addUser(user,dataset)
                if err == 1:
                    print("E' successo un imprevisto. Riprovare!")
                    continue
                users_in_dt.append(user)
                num_label = num_label + 1
                print("Sono presenti {} utenti nel dataset.".format(num_label))
            else:
                print("Utente già presente nel dataset!")
        elif answer == '3':
            classifier = tfidf = count_vect = None
            if num_label > 0:
                print("Sono presenti questi utenti:\n",users_in_dt)
                user = input("Indica un username da eliminare: ")
                dataset = dropUser(user,dataset,users_in_dt)
                users_in_dt.remove(user)
                num_label = num_label - 1
                print("Sono presenti {} utenti nel dataset.".format(num_label))
            else:
                print("Il dataset è vuoto! Nessun utente da eliminare!")
        elif answer == '4':
            print(users_in_dt)
        elif answer == '5':
            printDataset(dataset)
        elif answer == '6':
            if num_label > 1:       
                classifier, tfidf, count_vect = chooseClassifier(dataset,num_label)
                display(eli5.show_weights(classifier, vec=count_vect, top=40,target_names=classifier.classes_))
            else:
                print("Devi avere caricato i tweet di almeno 2 utenti diversi.\nAttualmente ve ne sono: {}".format(num_label))
        elif answer == '7':
            if classifier != None and num_label > 1:
                classify(classifier, tfidf, count_vect, dataset, api)
            if classifier == None:
                print("Devi prima scegliere un classificatore!")
            if num_label < 2:
                print("Devi avere caricato i tweet di almeno 2 utenti diversi.\nAttualmente ve ne sono: {}".format(num_label))
        elif answer == '8':
            break

"""SEZIONE DELLA PREDIZIONE DEI TWEET FORNITI E GENERAZIONE TABELLE DI PROBABILITA'"""


def classify(clas, tfidf, vect, dataset, api):
    while True:
        dt = None
        print("\n\nSCEGLI COME PREFERISCI FORNIRE IL TWEET DA CLASSIFICARE")
        print("1) Inserimento Manuale")
        print("2) Download da Tweet indicando un utente")
        print("3) Tweet presente all'interno al Dataframe")
        print("4) Ritornare al menù principale")
        print("[Default: 3]")
        answer = input("La tua scelta: ")
        
        if answer == '1':
            dt = manualTweets()
        elif answer == '2':
            if api != None:
                dt = apiTweets(api)
            else:
                print("Non hai inizializzato le API. Opzione non disponibile!")
                continue
        elif answer == '4':
            return
        else:
            dt = datasetTweets(dataset)
        
        clean_tweets = cleanTweet(dt['text'].tolist())
        count_tweets = vect.transform(clean_tweets)
        tf_tweets = tfidf.transform(count_tweets)
        
        Probas_x = pd.DataFrame(clas.predict_proba(tf_tweets), columns = clas.classes_)
        joined_x = dt
        joined_x[clas.classes_] = Probas_x
        joined_x = joined_x.reset_index(drop=True)
        print(joined_x)
		
        for x in dt['text'].tolist():
            display(eli5.show_prediction(clas, doc=x, vec=vect, top=40,target_names=clas.classes_))
        

def manualTweets():
    tweets = []
    
    print("Quanti tweet vuoi inserire?\n[Max 10, Default 1]\n")
    answer = input("La tua scelta: ")
    if answer.isdigit():
        if int(answer) > 10 or int(answer) < 1:
            answer = 1
        else:
            answer = int(answer)
    else:
        answer = 1
    
    for a in range(answer):
        print("\nScrivi un tweet:")
        tweet = input("Il tuo tweet: ")
        tweets.append(tweet)
        
    dt = pd.DataFrame()
    dt['text'] = tweets
    dt['handle'] = "you"
    
    return dt    

def apiTweets(api):
    tweets = []
    handles = []
    
    print("Da quanti utenti vuoi scaricare il tweet?\n[Max 5, Default 1]")
    answer = input("La tua scelta: ")
    if answer.isdigit():
        if int(answer) > 5 or int(answer) < 1:
            answer = 1
        else:
            answer = int(answer)
    else:
        answer = 1
        
    for a in range(answer):
        while True:
            try:
                print("\nScrivi un username da cui scaricare il tweet:")
                user = input("La tua scelta: ")
                statuses = api.GetUserTimeline(screen_name=user, count=1, include_rts=False)
                statuses = [_.AsDict() for _ in statuses]
                break
            except:
                print("Utente non trovato. Riprovare!")
        for item in statuses:
            if checkLenTweet(item['full_text']):
                tweets.append(item['full_text'])
                handles.append(user)
                print("Aggiunto un tweet di {}.\nTesto:\"{}\"\n\n".format(user, item['full_text']))
                break;
    
    dt = pd.DataFrame()
    dt['text'] = tweets
    dt['handle'] = handles
    
    return dt

def datasetTweets(dataset):
    tweets = []
    handles = []
    
    print("Quanti tweet vuoi prelevare dal dataset?\n[Max 10, Default 1]")
    answer = input("La tua scelta: ")
    if answer.isdigit():
        if int(answer) > 10 or int(answer) < 1:
            answer = 1
        else:
            answer = int(answer)
    else:
        answer = 1
    
    for a in range(answer):
        print("\nScrivi il numero della riga del dataset da cui prendere il tweet\n[Max {}, Default 0]:".format(len(dataset)-1))
        answer = input("La tua scelta: ")
        if answer.isdigit():
            if int(answer) > (len(dataset)-1) or int(answer) < 0:
                answer = 0
            else:
                answer = int(answer)
        else:
            answer = 0
        tweets.append(dataset.at[answer, 'text'])
        handles.append(dataset.at[answer, 'handle'])
        print("Prelevato il tweet #{} di {}!\n".format(answer, dataset.at[answer, 'handle']))   
        
    dt = pd.DataFrame()
    dt['text'] = tweets
    dt['handle'] = handles
    
    return dt










"""SEZIONE CLASSIFICATORI"""






#Stampa i plot dei dati
def printPlot(dataset,s=''):
    print(s)
    ans = input("Stampare il plot dei dati? [s,n] Default n. ")
    if ans == 's':
        pd.Series(dataset).value_counts().plot.bar()
        plt.show()

#Applica operazioni di preprocessing sul dataset
def clean(dataset):
    tweet_text = dataset['text'].values
    clean_text = cleanTweet(tweet_text)
    clean_df = pd.DataFrame(clean_text, columns=['x'])
    clean_df['label'] = dataset['handle'].values
    
    printPlot(clean_df['label'].tolist(),"Dataset")
    
    return clean_df

#Divide il dataset in TrS, TeS ed eventualmente VaS
def splitDataset(dataset,percent=0.25,knn=False,knn_percent=0.33):

    train_set, test_set, y_train_set, y_test_set  = train_test_split(dataset['x'].tolist(),dataset['label'].tolist(), test_size=percent)
    
    printPlot(y_train_set,'Training Set')
    
    if knn:
        train_set, vali_set, y_train_set, y_vali_set  = train_test_split(train_set,y_train_set, test_size=knn_percent)
        printPlot(y_vali_set,'Validation Set')
            
    printPlot(y_test_set,'Test Set')
    
    if knn:
        return train_set, y_train_set, test_set, y_test_set, vali_set, y_vali_set
    else:
        return train_set, y_train_set, test_set, y_test_set

#Regressore Logistico
def LRegr(dataset, num_label, features=20000, tdf=True):
    
    train_set, y_train_set, test_set, y_test_set = splitDataset(dataset)
    
    count_vect = CountVectorizer(ngram_range=N_GRAMS, max_features=features, max_df=STOP_WORDS_H, min_df=STOP_WORDS_L, strip_accents='unicode', stop_words=list_stop_words)
    tfidf = TfidfTransformer(use_idf=tdf)

    
    if num_label > 2:
        log = LogisticRegression(solver='liblinear', multi_class='ovr')
    else:
        log = LogisticRegression(solver='liblinear')
        
    x_train_counts = count_vect.fit_transform(train_set)
    x_train = tfidf.fit_transform(x_train_counts)
    
    x_test_counts = count_vect.transform(test_set)
    x_test = tfidf.transform(x_test_counts)

    log.fit(x_train, y_train_set)

    y_train_preds = log.predict(x_train)
    y_test_preds = log.predict(x_test)

    print("F1 training scores: {:0.2f}".format(f1_score(y_train_set,y_train_preds,average='weighted')))
    print("F1 test scores: {:0.2f}".format(f1_score(y_test_set,y_test_preds,average='weighted')))
    
    return log, tfidf, count_vect

#KNN
def KNN(dataset, features=750, tdf=True, k=0):
    
    train_set, y_train_set, test_set, y_test_set, vali_set, y_vali_set = splitDataset(dataset, knn = True)
    
    count_vect = CountVectorizer(ngram_range=N_GRAMS, max_features=features, max_df=STOP_WORDS_H, min_df=STOP_WORDS_L, strip_accents='unicode', stop_words=list_stop_words)
    tfidf = TfidfTransformer(use_idf=tdf)
    
    x_train_counts = count_vect.fit_transform(train_set)
    x_train = tfidf.fit_transform(x_train_counts)
    
    x_vali_counts = count_vect.transform(vali_set)
    x_vali = tfidf.transform(x_vali_counts)
    
    x_test_counts = count_vect.transform(test_set)
    x_test = tfidf.transform(x_test_counts)
      
    
    if k == 0:
        print("Valuteremo ora che K assegnare per massimizzare le performances.")
        best_score = 0
        best_k = 0

        for k_value in range(1,11):
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(x_train, y_train_set)
            y_vali_preds = knn.predict(x_vali)
            print("{} - F1 Validation Score: {:0.2f}".format(k_value,f1_score(y_vali_set,y_vali_preds,average='weighted')))
            if f1_score(y_vali_set,y_vali_preds,average='weighted') > best_score:
                best_score = f1_score(y_vali_set,y_vali_preds,average='weighted')
                best_k = k_value

        print("Il miglior K è {}.".format(best_k))
    else:
        k_value = k
        
    knn = KNeighborsClassifier(n_neighbors=k_value)

    knn.fit(x_train, y_train_set)

    y_train_preds = knn.predict(x_train)
    y_test_preds = knn.predict(x_test)

    print("F1 training scores: {:0.2f}".format(f1_score(y_train_set,y_train_preds,average='weighted')))
    print("F1 test scores: {:0.2f}".format(f1_score(y_test_set,y_test_preds,average='weighted')))
    return knn, tfidf, count_vect

#Multinomial Naive Bayes
def MNB(dataset, features=20000, tdf=True):
    
    train_set, y_train_set, test_set, y_test_set = splitDataset(dataset)
    
    count_vect = CountVectorizer(ngram_range=N_GRAMS, max_features=features, max_df=STOP_WORDS_H, min_df=STOP_WORDS_L, strip_accents='unicode', stop_words=list_stop_words)
    tfidf = TfidfTransformer(use_idf=tdf)

    x_train_counts = count_vect.fit_transform(train_set)
    x_train = tfidf.fit_transform(x_train_counts)
    
    x_test_counts = count_vect.transform(test_set)
    x_test = tfidf.transform(x_test_counts)

    nb = MultinomialNB()
    nb.fit(x_train, y_train_set)

    y_train_preds = nb.predict(x_train)
    y_test_preds = nb.predict(x_test)

    print("F1 training scores: {:0.2f}".format(f1_score(y_train_set,y_train_preds,average='weighted')))
    print("F1 test scores: {:0.2f}".format(f1_score(y_test_set,y_test_preds,average='weighted')))
    
    return nb, tfidf, count_vect

#Permette di scegliere tra i vari un classificatore
def chooseClassifier(dataset,num_label):
    answer = input("Indicare quale classificatore scegliere:\n1) Logistic Regressor\n2) Multinomial Naive Bayes\n3) KNN\nLa tua scelta è:[Default 1] ")
    
    if answer == '3':
        return KNN(clean(dataset))
    elif answer == '2':
        return MNB(clean(dataset))
    else:
        return LRegr(clean(dataset),num_label)
    
    
    
    
    
    
    
    
"""SEZIONE MINING E DATASET"""






#Implementa il mining dei tweet. Controlla se esiste un file .csv di quell'utente ed eventualmente chiede
#se sovrascrivere con una nuova ricerca di tweet
def mine_tweets(username,api,number_tweets=3000):
    miner = TweetMiner(api)
    try:
        user_df = pd.read_csv('../Data/{}.csv'.format(username))
        notExists = False
    except:
        notExists = True

    ans = ''

    if notExists == False:
        while True:
            ans = input("Trovato un file contenente i tweet dell'utente cercato. Vuoi sovrascrivere il contenuto di questo file? [s/n] ")
            if(ans == 's' or ans == 'n'):
                break
    if api == None:
        if ans == 's':
            print("Non hai inizializzato le API. Opzione non disponibile!\nUseremo il file presente in memoria.")
            return 0, api, username
        
        if notExists == True:
            print("Non hai inizializzato le API. Opzione non disponibile!\nVuoi inizializzarle? (s,n) [Default n]\n")
            answer = input('La tua scelta: ')
            if answer == 's':
                api = setAPI()
                return mine_tweets(username,api)
            else:
                print("Se non puoi inizializzarle, ti consigliamo di usare i dati degli utenti presenti in memoria.")
                return 1, api, username

    if ans == 's' or notExists == True:
        try:
            user_tweets = miner.mine_user_tweets(user=username)
            username = user_tweets[0]['handle']
        except Exception as e:
            print(e)
            print("Username non valido! Riprovare con un altro username!")
            return 1, api, username
        user_df = pd.DataFrame(user_tweets)
        user_df.to_csv('../Data/{}.csv'.format(username))
        print("Dati sovrascritti nel percorso '../Data/{}.csv!'".format(username))
    else:
        print("Useremo i dati contenuti nel file salvato precedentemente!")
    return 0, api, username
    
#Stampa la lista dei dataframe salvati in memoria
def printDF():
    import os
    fileList = os.listdir("../Data")
    try:
        fileList =  [re.search('[A-Za-z]*[^.csv]', x).group(0) if x != '.csv' else '' for x in fileList]
    except Exception as e:
        print(e)
    print(fileList)
    
#Restituisce la lista dei dataframe salvati in memoria
def getDF():
    import os
    fileList = os.listdir("../Data")
    try:
        fileList =  [re.search('[A-Za-z]*[^.csv]', x).group(0) if x != '.csv' else '' for x in fileList]
    except Exception as e:
        print(e)
    return(fileList)

#Aggiunge un utente (se esiste) nel Dataset
def addUser(username,dataset):
    try:
        user_df = pd.read_csv('../Data/{}.csv'.format(username))
        dataset = pd.concat([dataset, user_df], axis=0, sort=True)
        dataset = dataset.sort_values(by='retweet_count')
        dataset = dataset.drop(columns=['Unnamed: 0'])
        dataset = dataset.reset_index(drop=True)
        print("Aggiunto l'utente {}!".format(username))
    except Exception as e:
        print("Errore nell'aggiunta dell'utente!")
        print(e)
        return dataset, 1
    return dataset, 0

#Rimuove un utente (se presente) dal Dataset
def dropUser(username,dataset):
    print("Rimosso l'utente {}!".format(username))
    dt = dataset[dataset['handle'] != username]
    dt = dt.reset_index(drop=True)
    return dt

#Mostra una sezione del Dataset
def printDataset(dataset, low_lim = 0, high_lim = 0, user = None):
    if high_lim <= low_lim:
        high_lim = len(dataset)-1        
    if user == None:
        display(dataset[low_lim:high_lim])
    else:
        display(dataset[dataset['handle'] == user])
        
        
        
        
        
        
        
        
        
        

"""METODI DI SUPPORTO AL MINING E AD ALTRE OPERAZIONI VARIE"""

#TweetMiner di Mike Roman

class TweetMiner(object):

    
    def __init__(self, api, result_limit = 200, max_tweets=3000): #result_limit è il numero di tweet che scaricheremo ad ogni interazione con le api di Twitter
        
        self.api = api        
        if result_limit < 200: #Non può essere più di 200. 
            self.result_limit = result_limit
        else:
            self.result_limit = 200
        self.max_tweets = max_tweets
        

    def mine_user_tweets(self, user="", mine_retweets=False, no_replies=False, max_pages=40):
        import datetime
        
        data           =  []
        last_tweet_id  =  False
        page           =  1
        
        while len(data) < self.max_tweets and page <= max_pages:   
            if last_tweet_id: #Serve per evitare che si riprendano dei tweet già letti. Inizializziamo max_id con un id più piccolo dell'ultimo minato.
                statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.result_limit, max_id=last_tweet_id-1, include_rts=mine_retweets, exclude_replies=no_replies)
                statuses = [ _.AsDict() for _ in statuses]
            else:
                statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.result_limit, include_rts=mine_retweets, exclude_replies=no_replies)
                statuses = [_.AsDict() for _ in statuses]
                
            for item in statuses:
                # Si usa try except perchè quando retweets = 0 si riceve un errore (GetUserTimeline fallisce a creare una key, 'retweet_count')
                try:
                    mined = {
                        'tweet_id':        item['id'],
                        'handle':          item['user']['screen_name'],
                        'retweet_count':   item['retweet_count'],
                        'text':            item['full_text'],
                        'mined_at':        datetime.datetime.now(),
                        'created_at':      item['created_at'],
                    }
                
                except:
                    mined = {
                        'tweet_id':        item['id'],
                        'handle':          item['user']['screen_name'],
                        'retweet_count':   0,
                        'text':            item['full_text'],
                        'mined_at':        datetime.datetime.now(),
                        'created_at':      item['created_at'],
                    }
                
                last_tweet_id = mined['tweet_id']
                
                if checkLenTweet(mined['text']):
                    data.append(mined)
            page += 1
            print(len(data))
            
        return data
    
#Ripulisce un singolo tweet
def checkLenTweet(raw_data):
    data = txt.replace_urls(raw_data,"")
    data = txt.replace_emails(data,"")
    data = txt.replace_emojis(data,"")
    data = txt.replace_user_handles(data,"")
    data = txt.replace_phone_numbers(data,"")
    data = txt.replace_numbers(data,"")
    data = txt.replace_currency_symbols(data,"")
    data = txt.replace_hashtags(data,"")
    if len(data) < 28: #Lunghezza media di un tweet
        return False
    else:
        return True
    
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

#Gestisce l'accesso alle API di Twitter
def getAPI(ck, cs, atk, ats):
    
    twitter_keys = {
        'consumer_key':        ck,
        'consumer_secret':     cs,
        'access_token_key':    atk,
        'access_token_secret': ats
    }

    try:
        api = twitter.Api(
            consumer_key         =   twitter_keys['consumer_key'],
            consumer_secret      =   twitter_keys['consumer_secret'],
            access_token_key     =   twitter_keys['access_token_key'],
            access_token_secret  =   twitter_keys['access_token_secret'],
            tweet_mode = 'extended'
        )
        
        api.GetFriends() #Serve per far fallire in caso di non autenticazione
        print("API inizializzate con successo!")
        return api
    except Exception as e:
        print(e)
        print("Errore nell'accesso alle API! Riprovare!")
        return None

#Inizializza le API
def setAPI():
    print("\nINIZIALIZZA LE API. Ti chiederemo di fornire le chiavi di accesso alle API di TWITTER.\n")
    ck = input('consumer_key: ')
    cs = input('consumer_secret: ')
    atk = input('access_token_key: ')
    ats = input('access_token_secret: ')
    return getAPI(ck, cs, atk, ats)

#Crea la cartella da cui cercare i tweet
def makeData():
    import os
    print("\n\nCONTROLLO SE ESISTE LA CARTELLA DATA")
    try:
        os.mkdir('../Data')
        print("Cartella creata.")
    except:
        print("Cartella esistente.")
        
        
        
        
"""AVVIO DEL PROGRAMMA"""



main()