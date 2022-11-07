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

fig = plt.figure()
plt.rcParams['figure.figsize'] = (10, 10)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'white',
                      colormap='magma',
                      max_words = 100, 
                      stopwords = stopwords ,
                      width = 1200,
                      height = 800,
                     random_state = 30).generate(str(df['City']))


plt.title('Wordcloud for Station Names', fontsize = 25)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.show()


fig = plt.figure()
plt.rcParams['figure.figsize'] = (25, 10)
sns.barplot(df['City'],df['AQI'],palette='magma')
plt.title('Station Vs AQI', fontsize = 40)
plt.xlabel('Station', fontsize = 20)
plt.ylabel('AQI', fontsize = 20)
plt.show()


fig = plt.figure()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),cmap='viridis',annot=True)
plt.show()


fig = plt.figure()
monthly_sales = df.groupby(['City'])['AQI'].sum()
monthly_sales.plot()
plt.xlabel('StationId')
plt.ylabel('AQI')
plt.title('StationId AQI')
plt.show()


### Graph Plotting between entities:
fig = plt.figure()
#plt.subplot(3,2,1)
plt.bar(x2[:25],y1[:25],color='green',width=5)
plt.title('Air Quality Index')
plt.xlabel('AQI')
plt.ylabel('PM2.5')

fig = plt.figure()
#plt.subplot(3,2,2)
plt.bar(x3[:25],y1[:25],color='blue',width=5)
plt.title('Air Quality Index')
plt.xlabel('AQI')
plt.ylabel('PM10')
#plt.show()


fig = plt.figure()
#plt.subplot(3,2,4)
ax=sns.barplot(x14,y1)
ax.set(xlabel='Datetime',ylabel='AQI')
plt.title('Index');
plt.show()
