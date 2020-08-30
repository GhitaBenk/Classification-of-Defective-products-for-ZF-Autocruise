#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pickle
import pandas as pd
import numpy as np
import copy
import string
import sys 
np.set_printoptions(threshold=sys.maxsize)

def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


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


# In[54]:


data_total_float


# In[56]:


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


# In[57]:


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


# In[24]:


df0


# In[15]:


df0


# In[4]:


from keras import models
from keras import layers
from keras.optimizers import Adam
from sklearn.model_selection import  train_test_split

test_network = models.Sequential()
#test_network.add(layers.Dense(512, activation='relu', input_shape=(7,))) 
#test_network.add(layers.Dense(16, activation='relu'))
test_network.add(layers.Dense(16, activation = 'relu' ))
#test_network.add(layers.Dense(6, activation = 'relu' ))
test_network.add(layers.Dense(4, activation='sigmoid'))

#test_network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
test_network.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='categorical_crossentropy', metrics=['accuracy'])

data_train,data_valid, label_train, label_valid = train_test_split(data_total_float,data_one_hot,test_size = 0.2,random_state = 0)

"""
data_train = data_total_float[0:5500]
data_valid = data_total_float[5500:6001]

label_train = data_one_hot[0:5500]
label_valid = data_one_hot[5500:6001]
"""

#test_network.fit(data_used_normed, data_one_hot, epochs=10, batch_size=128)
his = test_network.fit(data_train, label_train, epochs=50, batch_size=128 , validation_data=(data_valid, label_valid), verbose=1)

test_network.save('test_network.h5') 
test_network.summary()


# In[5]:


#don't need run
# After training the model, we can save the model that we trained
test_network.save('model.h5') 
test_network.save_weights('my_model_weights.h5')


# In[6]:


# We can use the model to predict.

from keras.models import load_model
model = load_model('model.h5')


from keras.models import model_from_json 
json_string = model.to_json() 
model = model_from_json(json_string) 
#print(json_string)


# In[7]:


# compile the model
model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='categorical_crossentropy'i, metrics=['accuracy'])


# In[8]:


# test the model and to predict, know the Accuracy
loss,accuracy = test_network.evaluate(data_valid, label_valid, batch_size=1)
print(loss)
print(accuracy)


# In[9]:


output = test_network.predict(data_valid)
print(output.argmax(axis=1))


# In[10]:


# list all data in history
print(his.history.keys())

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(his.history['acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(his.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # SVM

# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(data_total_float,label,test_size = 0.2,random_state = 0)
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(classification_report(y_test, y_pred)) 
y_train_pred = svclassifier.predict(X_train)
print(classification_report(y_train, y_train_pred)) 


# In[71]:


X_train.T[:2].shape


# In[76]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X_train)
pca_2d = pca.transform(X_train)
import pylab as pl
for i in range(0, pca_2d.shape[0]):
    if y_train[i] == 0:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    marker='+')
    elif y_train[i] == 1:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    marker='o')
    elif y_train[i] == 2:
        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    marker='*')
    elif y_train[i] == 3:
        c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='y',    marker='.')
        
pl.legend([c1, c2, c3,c4], ['FUITE', 'VISSAGE','CTRL NOK' ,  'TRAVAIL NORMAL'])
pl.title('Iris training dataset with 3 classes and    known outcomes')
pl.show()


# In[83]:


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pylab as pl
import numpy as np
X_train, X_test, y_train, y_test =   train_test_split(data_total_float,label , test_size=0.10, random_state=111)
pca = PCA(n_components=2).fit(X_train)
pca_2d = pca.transform(X_train)
svmClassifier_2d =   svm.LinearSVC(random_state=111).fit(   pca_2d, y_train)
for i in range(0, pca_2d.shape[0]):
    if y_train[i] == 0:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    s=50,marker='+')
    elif y_train[i] == 1:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    s=50,marker='o')
    elif y_train[i] == 2:
        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    s=50,marker='*')
    elif y_train[i] == 3:
        c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='y',    marker='.')
        
pl.legend([c1, c2, c3,c4], ['FUITE', 'VISSAGE','CTRL NOK' ,  'TRAVAIL NORMAL'])
x_min, x_max = pca_2d[:, 0].min() - 1,   pca_2d[:,0].max() + 1
y_min, y_max = pca_2d[:, 1].min() - 1,   pca_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
Z = svmClassifier_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
Z = Z.reshape(xx.shape)
pl.contour(xx, yy, Z)
pl.title('Support Vector Machine Decision Surface')
pl.axis('off')
pl.show()


# In[64]:


get_ipython().system('pip install mlxtend')


# In[60]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
# Take the first two features. We could avoid this by using a two-dim dataset


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X_train, y_train) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_train[:, 0], X_train[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=label, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


# In[84]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X_train, X_test, y_train, y_test = train_test_split(data_total_float, label, test_size = 0.20)

clf = RandomForestClassifier(n_estimators=100,criterion = 'gini'
                             )
clf.fit(X_train, y_train)


# In[85]:


clf.score(X_train, y_train)


# In[86]:


clf.score(X_test, y_test)


# In[87]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred)) 


# In[91]:


from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap=plt.cm.Blues)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[95]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion = 'entropy',
            random_state = 100,max_depth= 4 , min_samples_leaf=14)
    
clf.fit(X_train, y_train)
 
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred)) 


# In[106]:


from sklearn import tree
plt.figure(figsize = (20,10))
tree.plot_tree(clf.fit(df, label) , class_names = d[0] ,fontsize = 10 , filled = True , label='all', rounded=True ) 
plt.show()


# In[30]:


df = pd.DataFrame(data_total_float)


# In[46]:


l = pd.DataFrame(label)
a = l.replace(0,'FUITE')
b = a.replace(1,'VISSAGE')
c = b.replace(2,'CTRL NOK')
d = c.replace(3,'TRAVAIL NORMAL')


# In[51]:


d[0]


# In[ ]:




