import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import missingno as msno
from sklearn.impute import KNNImputer
# Autoreg, autocorrolationand time series tools...

from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic


plt.style.use('seaborn-whitegrid')

from termcolor import colored

import os

        
city_day = pd.read_csv('city_day.csv').sort_values(by = ['Date', 'City'])
print(list(city_day.columns))
print(city_day.head(2))
print(city_day.info())
city_day.Date = city_day.Date.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d'))
city_day = city_day.sort_values(by = 'Date')
print('Date starts from {}, and ends in {}'.format(city_day.Date.min().strftime('%Y-%m-%d'), city_day.Date.max().strftime('%Y-%m-%d')))
city_day.corr().AQI.sort_values(ascending = False)
# adding all the features with corr less than 0.4

city_day['B_X_O3_NH3'] = city_day['Benzene'] +\
city_day['Xylene'] + city_day['O3'] + city_day['NH3']

city_day['ParticulateMatters'] = city_day['PM2.5'] + city_day['PM10']

corr_with_AQI = city_day.corr().AQI.sort_values(ascending = False)


print(corr_with_AQI)
# from here we can see: we can impute values with linear
# interpolation for the ones that have high value of corr


# how much is the average amount of pollution in each city stations
most_polluted = city_day[['City', 'AQI', 'PM10', 'CO']].groupby(['City']).mean().sort_values(by = 'AQI', ascending = False)
print(most_polluted)

most_polluted = city_day[['City', 'AQI', 'PM10', 'CO']].groupby(['City']).mean().sort_values(by = 'AQI', ascending = False)

cities = most_polluted.index
params = most_polluted.columns

def first_date(city, parameter):
    df = city_day[(city_day.City == city)]
    df = df[df[parameter].notnull()]
    if len(df) != 0:
        return df.iloc[0].Date.strftime('%Y-%m-%d')
    else: return('no_measurement')
        
        
for city in cities:
    #print(colored('city: ', 'green'), city)
    for param in params:
      #  print('param: ', param)
        most_polluted.loc[city, str(param) + '_date'] = first_date(city, param)
        
print(most_polluted)
#plt.figure(figsize=(10,10))
plt.style.use('seaborn-whitegrid')
f, ax_ = plt.subplots(1, 3, figsize = (15,15))

bar1 = sns.barplot(x = most_polluted.AQI,
                   y = most_polluted.index,
                   palette = 'Reds_r',
                   ax = ax_[0]);

bar1 = sns.barplot(x = most_polluted.PM10,
                   y = most_polluted.index,
                   palette = 'RdBu',
                   ax = ax_[1]);

bar1 = sns.barplot(x = most_polluted.CO,
                   y = most_polluted.index,
                   palette = 'RdBu',
                   ax = ax_[2]);

titles = ['AirQualityIndex', 'ParticulateMatter10', 'CO']
for i in range(3) :
    ax_[i].set_ylabel('')   
    ax_[i].set_yticklabels(labels = ax_[i].get_yticklabels(),fontsize = 14);
    ax_[i].set_title(titles[i])
    f.tight_layout()
plt.show()


# Sum of pollution
import plotly.express as px
#plt.figure(figsize=(10,10))
df = city_day.drop(columns = ['Date', 'AQI_Bucket', 'AQI']).groupby('City').sum().reset_index()
fig = px.treemap(pd.melt(df, id_vars = 'City'), path=['City','variable'],values=pd.melt(df, id_vars = 'City')['value'],title = 'Cities and the proportion of pollution in each')
fig.show()


city_day['Year_Month'] = city_day.Date.apply(lambda x : x.strftime('%Y-%m'))

df = city_day.groupby(['Year_Month']).sum().reset_index()

# let's only see those that are important to the AQI
# otherwise we will have a messy plot

metrices = corr_with_AQI[corr_with_AQI>0.5].index

#plt.figure(figsize=(10,10))
plt.style.use('seaborn-whitegrid');
fig, ax_ = plt.subplots(figsize=(20,10));
df = city_day.groupby(['Year_Month']).sum().reset_index()
for col in metrices:
    x = df['Year_Month']
    y = df[col]
    ax_.plot_date(x ,y ,label=col, linestyle="-");
ax_.set_xticklabels(df['Year_Month'], rotation=85);
ax_.legend();
plt.show()


city_day['Month'] = city_day.Date.dt.month
city_day['Year'] = city_day.Date.dt.year

index = 'Month'
df = city_day.groupby([index]).sum().reset_index()

plt.style.use('seaborn-whitegrid');
fig, ax_ = plt.subplots(figsize=(21,8));


for i, col in enumerate(metrices):
    x = df[index]
    y = df[col]
    plot = sns.lineplot(x ,y );
plot.set_xticklabels(df[index], );
ax_.set(xlabel='Metrics', ylabel='Months');
leg = plot.legend(title='legends', loc='upper left', labels=metrices, fontsize = 11);
plt.show()



