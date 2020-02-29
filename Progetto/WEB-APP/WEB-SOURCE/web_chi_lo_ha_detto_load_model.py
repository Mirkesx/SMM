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
import warnings
warnings.simplefilter('ignore')

#MAIN
import joblib
import sys

message = sys.argv[1]#'Lasciatemi cantare con la chitarra in mano lasciatemi cantare sono un italiano Buongiorno Italia gli spaghetti al dente e un partigiano come Presidente autoradio sempre nella mano destra'
type_mes = sys.argv[2]#'POLITICO'

filename ='../WEB-DATA/MODELS/'+sys.argv[2]+'.sav'
clas = joblib.load(filename)
filename ='../WEB-DATA/VECTORS/'+sys.argv[2]+'.sav'
vect = joblib.load(filename)
    
mess_vect = vect.transform([message])

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