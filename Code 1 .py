#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[4]:


files = ['2019_S0'+str(i) for i in range(1,9)]+['2019_S'+str(i) for i in range(10,41)]
onlyfiles = []

for i in range(len(files)) :
     
    onlyfiles.append([f for f in listdir('C:\\Users\\benki\\Desktop\\NOKAssemblage\\'+files[i]) ])
    
p = 0    
T = [0 for k in range(60)] 
  
for j in range(len(onlyfiles))  :
    
    for fichier in onlyfiles[j]:
        
        
        f = open('C:\\Users\\benki\\Desktop\\NOKAssemblage\\'+files[j]+'\\'+fichier, "r")
        lines = f.readlines()
        if len(lines) == 105 :
            
            p += 1
            
        
            features_indices = [10,24,25,26,27,28,31,33,34]+[i for i in range(48,99)]
            
            for k in range(len(features_indices)):
            
                if len(lines[features_indices[k]].split(';')[1]) ==  0:
                    
                        T[k] = T[k] + 1
    
L = [0 for k in range(60)]        
for k in range(len(T))   :
    L[k] = (T[k]*100)/5630
L #représente le pourcentage avec lequel chaque paramétre n'est pas présent dans les données
#chaque element dans la liste T repésente combien de fois le paramétre  n'est pas présent pour tout les fichier

#len(features_indices)


# In[5]:


def getfeatures(features):
    right_features = []
    for i in range(len(L)) :
        if L[i] < 36 :
            right_features.append(features[i])
    right_features.remove(right_features[0])
    return right_features


# In[6]:


def Matrix(features):
    
    
    
    X =[]   
    for j in range(len(onlyfiles))  :
        
        for fichier in onlyfiles[j]:
            
            f = open('C:\\Users\\benki\\Desktop\\NOKAssemblage\\'+files[j]+'\\'+ fichier, "r")
            lines = f.readlines()
            if len(lines) == 105 :
                L = []
                for i in features :

                    L.append(lines[i].split(';')[1])
            X.append(L)
    return X

        


# In[7]:


import string

l =   Matrix(getfeatures(features_indices))
alphabet = string.ascii_lowercase+string.ascii_uppercase    
for i in range(len(l)):
    for j in range(len(l[0])):
        for k in l[i][j]:
            if k in alphabet:
                l[i][j] = l[i][j].replace(k,'')

                    
                    
                    


# In[8]:


for i in range(len(l)):
    for j in range(len(l[0])):
        if l[i][j] != '':
            l[i][j] = float(l[i][j])
        
        else :
            l[i][j] = 0
        


# Récupération de Y

# In[9]:


Y_lab = Matrix([10])
i = len(Y_lab)-1
while i > 0 :

    if Y_lab[i] == ['']:
        Y_lab.remove(Y_lab[i])
        l.remove(l[i])

    i -= 1


Y_lab


# In[10]:


categories = Y_lab.copy()


# In[11]:


A = [['FUNCTIONAL TEST'],['TEST ATEQ NOK']]
B = [['CLIPSAGE NOK'],['CODE PCB INCOMPATIBLE'], ['DELTA T PLASMA REAR H'],['ABSENCE REAR HOUSING'],['RADAR FERME OU ABSENT'],['CTRL NOK PATE THERMIQUE'],['CTRL NOK SILICONE'],['CTRL PINS NOK'],['CYCLE NON FINI'],['CYCLE NON TERMINE'],['DEF PATE THERMIQUE PTA'],['DEF PATE THERMIQUE PTB'],['DEFAUT PLASMA CONNECTEUR'],['DEFAUT PLASMA FRONT HOUS'],['DEFAUT PLASMA RADAR PT A'],['DEFAUT PLASMA RADAR PT B'],['DEFAUT PLASMA REAR HOUS.'],['DEFAUT PLASMA REAR HOUSING'],['DEFAUT PRISE AU LASER'],['DEFAUT SILICONE'],['DELTA T PATE THERMIQUE'],['DELTA T SILICONE'],['DMC INCOHERENT RECETTE'],['DMC NUM ACSN NOK'],['DMC NUM INTERNE NOK'],['GRAVAGE FACE CONNECTEUR'],['GRAVAGE FACE OPPOSEE'],['GRAVAGE FRONT HOUSING'],['INSERTION CONNECTEUR NOK'],['LECTURE CODE PCB NOK'],['PRESS FIT NOK'],['REAR HOUSING NON VIERGE'],['DEFAUT PATE THERMIQUE'],['REFLASH PCB NOK'],['VT L']]
C = [['VISSAGE 1 NOK'],['VISSAGE 2 NOK'],['VISSAGE 3 NOK'],['VISSAGE 4 NOK'],['VISSAGE 5 NOK'],['VISSAGE  6 NOK'],['VISSAGE  7 NOK'],['VISSAGE  8 NOK'],['VISSAGE  9 NOK'],['VISSAGE 10 NOK']]
for i in range(len(categories)) :
    if categories[i] in B :
         categories[i] = ['CTRL NOK']
    elif categories[i] in A :
         categories[i] = ['FUITE']
    elif  categories[i] in C :
         categories[i] = ['VISSAGE']


# In[77]:


categories


# In[12]:



from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(categories)
Y


# In[13]:


n = len(categories)
categories = [ categories[i][0] for i in range(n)]


# In[14]:


df = pd.DataFrame(l)
df['dep_var'] = categories


# In[15]:


df = df[(df.iloc[:,[3,4,5]].T != 0).any()]
df


# In[16]:



from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(df['dep_var'])
Y


# In[127]:



df.to_pickle('Processed_Data_2.pkl')


# In[17]:


X , y = np.array(df.iloc[:,[0,1,2,3,4,5]]), Y
X


# In[16]:


from pandas.plotting import scatter_matrix
from matplotlib import cm
feature_names = [0, 1, 2]
X = df[feature_names]
y = df['y']
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(df,alpha = 0.2,figsize = (20,20),c = y)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('products_scatter_matrix')


# In[484]:


df


# # PCA

# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Functions for nice plot

# In[69]:


def plot_corr(corr):
    # https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap)


# In[18]:


sns.set_style('darkgrid')
plt.rc('figure', figsize=[8, 6])


# # Preprocessing

# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


#X = df.values
#print(X)


# In[19]:


std_scale = StandardScaler().fit(X)
X_scaled = std_scale.transform(X)


# In[20]:


# Mean and variance per columns (axis=0) before scaling
np.mean(X, axis=0), np.var(X, axis=0)


# In[21]:


# After scaling
# The mean is close to 0 (not exactly due to the limited precision of computers)
# and the variance is 1. Scaling the variance to 1 for every features allows us
# to give the same importance to the variations of each features, even though they
# may have different amplitudes in the original dataset.
np.mean(X_scaled, axis=0), np.var(X_scaled, axis=0)


# # PCA with sklearn

# In[22]:


from sklearn.decomposition import PCA


# In[23]:


# Fit PCA only on *active* variables (months)
pca = PCA().fit(X_scaled)


# In[24]:


# 99% of the variance is explained by the first two components
# Project and plot the data the on first two components
X_proj = pca.transform(X_scaled)
X_proj


# In[25]:


_, axes = plt.subplots(ncols=4, figsize=(16,4))
for i, (ax, col) in enumerate(zip(axes, ['lat', 'long', 'moy', 'amp'])):
    ax.scatter(X_proj[:,1], data[col])
    ax.set_title(f'2nd component vs {col}')


# In[418]:


plt.scatter(X_proj[:,0], X_proj[:,1])
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.xlim(-7, 7)
plt.ylim(-5, 5)

#for i in range(X_proj.shape[0]):
    #plt.text(X_proj[i,0], X_proj[i,1], df.index[i])


# In[210]:


#X_pca = X_proj[:, 0:3]
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=3).fit_transform(X_scaled)


# In[211]:


df_embedded = pd.DataFrame(X_embedded)
df_embedded['dep_var'] = categories
df_embedded.to_pickle('tSNR_Data.pkl')
df_embedded


# In[212]:


plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.xlim(-100, 100)
plt.ylim(-100, 100)


# # SVM

# In[175]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(classification_report(y_test, y_pred)) 
y_train_pred = svclassifier.predict(X_train)
print(classification_report(y_train, y_train_pred)) 


# In[ ]:





# In[ ]:


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
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


# ## Random Forest

# In[138]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

clf = RandomForestClassifier(n_estimators=100,criterion = 'gini'
                             )
clf.fit(X_train, y_train)


# In[139]:


clf.feature_importances_


# In[140]:


clf.score(X_train, y_train)


# In[141]:


clf.score(X_test, y_test)


# In[142]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred)) 


# In[147]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion = 'entropy',
            random_state = 100,max_depth= 4 , min_samples_leaf=14)
    
clf.fit(X_train, y_train)
 
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred)) 


# In[101]:


from sklearn import tree
plt.figure(figsize = (20,20))
tree.plot_tree(clf.fit(df.iloc[:,[0,1,2]], y),class_names = df['dep_var'],fontsize = 10) 
plt.show()


# # Plotting

# In[18]:


df['y'] = Y


# In[32]:


df.iloc[:,[1]].shape


# In[47]:


from mpl_toolkits import mplot3d

# Data for a three-dimensional line
#zline = np.linspace(0, 15, 1000)
#xline = df.iloc[:,[1]]
#yline = np.cos(zline)
#ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
for i in range(5):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    zdata = [0]*3592
    xdata = df.iloc[:,[i]].values
    ydata = df.iloc[:,[i+1]].values
    ax.scatter3D(xdata, ydata, zdata, c=Y , cmap='Greens');
    plt.xlabel('feature'+str(i))
    plt.ylabel('feature'+str(i+1))
    plt.show()


# In[169]:


for i in range(2,6):
    plt.scatter(df.iloc[:,[1]],df.iloc[:,[i]] ,c = b  )
    plt.xlabel('feature1')
    plt.ylabel('feature'+str(i))
    plt.show()


# In[174]:


A.hist(column = [0,1,2,3,4,5],figsize = (10,10) , by = df['dep_var'] , bins = 2)
plt.show()


# In[136]:


import plotly.express as px
px.parallel_coordinates(A, color = y,color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)


# In[498]:


get_ipython().system('pip install plotly')


# In[19]:


A = df.drop(columns = ['y']) 


# In[21]:


A.hist(column = [0,1,2,3,4,5],figsize = (10,10) )


# In[ ]:




