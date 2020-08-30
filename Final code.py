#!/usr/bin/env python
# coding: utf-8

# # Nettoyage du data 

# In[1]:


import pickle
import pandas as pd
import numpy as np
import copy
import string
import sys 
from sklearn.model_selection import  train_test_split

np.set_printoptions(threshold=sys.maxsize)

def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

#on importe la data
with open('Data.pkl', 'rb') as f:
    data = pickle.load(f)
data_total = np.array(copy.deepcopy(data))
defaut_row = pd.read_excel('example_data.xlsx')
defaut = np.array(defaut_row)
defaut = defaut[48:95,5:]
data_total = data_total[:,5:52]
alphabet = string.ascii_lowercase+string.ascii_uppercase    
numbers = np.array(['0','1','2','3','4','5','6','7','8','9','-','.']) 
data_total_float = np.zeros(shape=(6001,47))

#on remplace les valeurs vides avec les valeurs par défaut 
for i in range(6001):
    for j in range(47):
        if data_total[i][j] == '':
            data_total[i][j] = to_str(defaut[j][0])
        for k in data_total[i][j]:
            if k in alphabet:
                data_total[i][j] = data_total[i][j].replace(k,'')
            if k == ' ':
                data_total[i][j] = data_total[i][j].replace(k,'')
            if k not in numbers:
                data_total[i][j] = data_total[i][j].replace(k,'')
        try:
            data_total_float[i][j] = float(data_total[i][j])
        except:
            data_total_float[i][j] = float(0)


# In[2]:


#fonction qui permettra de de rendre les chaine de caractère un nombre réel
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


# In[3]:


#on regroupe les 37 classes en 4 classes principales 
df0 = pd.read_pickle('Data.pkl')
df1 = df0[0]
label = df1.replace('TEST ATEQ NOK',0)
label1 = label.replace('TRAVAIL NORMAL',3)
label = label1.replace(['VISSAGE 1 NOK','VISSAGE 2 NOK','VISSAGE 3 NOK','VISSAGE 4 NOK','VISSAGE 5 NOK','VISSAGE 6 NOK','VISSAGE  6 NOK','VISSAGE  7 NOK','VISSAGE  8 NOK','VISSAGE  9 NOK','VISSAGE 10 NOK','VISSAGE  10 NOK'],[1,1,1,1,1,1,1,1,1,1,1,1])

for i in label:
    if is_number(i):
        pass      
    else:
        label = label.replace(i,2)
        
label = np.array(label)
data_one_hot = np.array(pd.get_dummies(label))
data_one_hot.shape


# # Random forest

# In[4]:


#on utilise la méthode random forest pour la classification 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#on divise la datase en data d'entrainement et data de test 
X_train, X_test, y_train, y_test = train_test_split(data_total_float, label, test_size = 0.20)

clf = RandomForestClassifier(n_estimators=100,criterion = 'gini'
                             )
clf.fit(X_train, y_train)


# In[5]:


clf.score(X_train, y_train)


# In[6]:


clf.score(X_test, y_test)


# In[7]:


#on évolue la méthode en calculant l'accuracy
from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred)) 


# # Utilisation de l'alforithme pour prédire les classes

# In[12]:


#en executant cette cellule vous pourriez entrer le fichier et vous aurez la classe à laquelle il correspond
X = []
f = open(input(), "r")
lines = f.readlines()
if len(lines) == 105 :
    L = []
    for i in range(len(lines)) :

        L.append(lines[i].split(';')[1])
X.append(L)



df  = pd.DataFrame(X)








data_total = np.array(copy.deepcopy(X))
defaut_row = pd.read_excel('example_data.xlsx')
defaut = np.array(defaut_row)
defaut = defaut[48:95,5:]
data_total = data_total[:,5:52]
alphabet = string.ascii_lowercase+string.ascii_uppercase    
numbers = np.array(['0','1','2','3','4','5','6','7','8','9','-','.']) 
data_total_float = np.zeros(shape=(1,47))

#on remplace les valeurs vides avec les valeurs par défaut 
for j in range(47):
    if data_total[0][j] == '':
        data_total[0][j] = to_str(defaut[j][0])
    for k in data_total[0][j]:
        if k in alphabet:
            data_total[0][j] = data_total[0][j].replace(k,'')
        if k == ' ':
            data_total[0][j] = data_total[0][j].replace(k,'')
        if k not in numbers:
            data_total[0][j] = data_total[0][j].replace(k,'')
    try:
        data_total_float[0][j] = float(data_total[0][j])
    except:
        data_total_float[0][j] = float(0)


# In[13]:


df2 = df[0]
label = df2.replace('TEST ATEQ NOK',0)
label1 = label.replace('TRAVAIL NORMAL',3)
label = label1.replace(['VISSAGE 1 NOK','VISSAGE 2 NOK','VISSAGE 3 NOK','VISSAGE 4 NOK','VISSAGE 5 NOK','VISSAGE 6 NOK','VISSAGE  6 NOK','VISSAGE  7 NOK','VISSAGE  8 NOK','VISSAGE  9 NOK','VISSAGE 10 NOK','VISSAGE  10 NOK'],[1,1,1,1,1,1,1,1,1,1,1,1])

for i in label:
    if is_number(i):
        pass      
    else:
        label = label.replace(i,2)
        
label = np.array(label)


# In[14]:


clf.predict(data_total_float)


# In[ ]:




