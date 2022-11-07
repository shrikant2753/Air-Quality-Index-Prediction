from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn import svm
from tkinter import messagebox
from tkinter import *
import tkinter
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import joblib


# Dataset Link: https://www.kaggle.com/rohanrao/air-quality-data-in-india

warnings.filterwarnings('ignore')


df=pd.read_csv('city_day.csv')
#print(df)

df1=df.iloc[:,:14]
df1.fillna(0,inplace=True)      #it replaces all empty spaces with 0
#print(df1)

x1=df['City']
LE= preprocessing.LabelEncoder()
xg=LE.fit_transform(x1)
x1=xg

x2=df1['PM2.5']
x3=df1['PM10']
x4=df1['NO']
x5=df1['NO2']
x6=df1['NOx']
x7=df1['NH3']
x8=df1['CO']
x9=df1['SO2']
x10=df1['O3']
x11=df1['Benzene']
x12=df1['Toluene']
x13=df1['Xylene']

X=np.matrix([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])
X=np.transpose(X)
X=X[:1000,:14]

# Standardization of data of X:
scalar = StandardScaler()
X=scalar.fit_transform(X)

y=df['AQI_Bucket']
y.fillna('unknown',inplace=True)
y1=df['AQI']
#print(y)
y=LE.fit_transform(y)
y=y[:1000]
#print(y)

x14=df1['Date']
x14=x14[:100]

y1=y1[:100]


mdl2=KN(n_neighbors=41).fit(X,y)
print('Accuracy of KN-Neighbors is: ')
print(mdl2.score(X,y),'\n','\n')



mdl4=svm.SVC(kernel='rbf', C=1,gamma=1).fit(X, y)         
print('Accuracy of SVC is: ')
print(mdl4.score(X,y),'\n')
