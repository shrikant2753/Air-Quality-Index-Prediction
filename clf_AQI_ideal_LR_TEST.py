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


# save the model to disk
filename = 'LE.sav'
LE = joblib.load(filename)

# save the model to disk
filename = 'scalar.sav'
scalar = joblib.load(filename)

# save the model to disk
filename = 'mdl2.sav'
mdl2 = joblib.load(filename)

# save the model to disk
filename = 'mdl4.sav'
mdl4 = joblib.load(filename)


# *********************************************************************************
### INPUT FROM USER : UNCOMMENT THIS
##test1=input('enter StationId: ')
##C1=[test1]
##C1=LE.transform(C1)
##print('City:',C1)
##test2=float(input('enter PM2.5 value: '))
##test3=float(input('enter PM10 value: '))
##test4=float(input('enter NO value: '))
##test5=float(input('enter NO2 value: '))
##test6=float(input('enter NOx value: '))
##test7=float(input('enter NH3 value: '))
##test8=float(input('enter CO value: '))
##test9=float(input('enter SO2 value: '))
##test10=float(input('enter O3 value: '))
##test11=float(input('enter Benzene value: '))
##test12=float(input('enter Toluene value: '))
##test13=float(input('enter Xylene value: '))
##test=np.matrix([int(C1),test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13])
##print(test,np.shape(test))
##test=scalar.fit_transform(test)
# *********************************************************************************
### INPUT FROM HARDCODE : UNCOMMENT THIS
test1=input('enter StationId: ')
C1=[test1]
C1=LE.transform(C1)
print('City:',C1)
test=np.matrix([int(C1),104,148.5,1.93,23,13.75,9.8,0.1,15.3,117.62,0.3,10.4,0.23])
test=scalar.fit_transform(test)
# *********************************************************************************

k = mdl4.predict(test)
k=np.abs(k)
k=np.floor(k)

#k=np.argmax(k)
print('Predicted Using SVM: ' ,k)

if k==0:
    print('AQI is Good')
elif k==1:
    print('AQI is Moderate')
elif k==2:
    print('AQI is Poor')
elif k==3:
    print('AQI is Satisfactory')
elif k==4:
    print('AQI is Severe')
elif k==5:
    print('AQI is Unknown')
elif k==6:
    print('AQI is Very Poor')


k = mdl2.predict(test)
k=np.abs(k)
k=np.floor(k)

#k=np.argmax(k)
print('Predicted Using KNN: ' ,k)

if k==0:
    print('AQI is Good')
elif k==1:
    print('AQI is Moderate')
elif k==2:
    print('AQI is Poor')
elif k==3:
    print('AQI is Satisfactory')
elif k==4:
    print('AQI is Severe')
elif k==5:
    print('AQI is Unknown')
elif k==6:
    print('AQI is Very Poor')
