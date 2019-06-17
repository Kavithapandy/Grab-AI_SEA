
# coding: utf-8

# In[1]:

#Defining local path 
your_local_path="C:/Users/nlakshman/Desktop/Grab Challenge/Traffic Management/"


# In[2]:

#Importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[3]:

#Reading input file
traf_mgmt_file=your_local_path+'training.csv'
print(traf_mgmt_file)
traf_mgmt_data=pd.read_csv(traf_mgmt_file)
#Head of input
traf_mgmt_data.head()


# In[4]:

#Formatting day and timestamp fields to default datetime format
from datetime import datetime,date,time
traf_mgmt_data['date']=pd.to_datetime(traf_mgmt_data['day'],unit='D')
traf_mgmt_data['time']=pd.to_datetime(traf_mgmt_data['timestamp'],format='%H:%M').dt.time


# In[5]:

#Merging the formatted date and timestamp
traf_mgmt_data['datetim']=traf_mgmt_data.apply(lambda t : pd.datetime.combine(t['date'],t['time']),1)


# In[6]:

#Setting datetime as index and sorting
traf_mgmt_data=(traf_mgmt_data.sort_values(by=['datetim'])).set_index(['datetim'])
traf_mgmt_data.tail()


# In[7]:

#Converting Geohash to its respective Latitude and Longitude
import geohash2 as pgh
traf_mgmt_data['latlong']=traf_mgmt_data['geohash6'].apply(lambda x:pgh.decode(x))
traf_mgmt_data.tail()


# In[8]:

#Splitting the latitude and longitude
traf_mgmt_data[['lat','long']]=pd.DataFrame(traf_mgmt_data['latlong'].tolist(),index=traf_mgmt_data.index)
traf_mgmt_data.tail()


# In[9]:

#To visualize the demand variation with respect to time
traf_mgmt_grp=traf_mgmt_data.groupby(traf_mgmt_data.index).mean()
traf_mgmt_grp


# In[10]:

#Line Plot - Timestamp Vs Demand(mean)
plt.figure(figsize=(30,10))
plt.title('Datetime Vs Demand - Line plot',fontsize=25)
plt.xlabel('Datetime',fontsize=20)
plt.ylabel('Demand',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.plot(traf_mgmt_grp['demand'])


# In[11]:

#Observation: From the Line plot, its evident that there is no increasing or decreasing trend of demand with respect to datetime.
#However for a shorter duration(61 days), the trend may seem to be stationary
#Day18 has unusual demand drop - below 0.025


# In[12]:

#Scatter Plot - Timestamp Vs Demand(mean)
x=traf_mgmt_grp.index
y=traf_mgmt_grp['demand']
plt.figure(figsize=(30,10))
plt.title('Datetime Vs Demand - Scatter plot',fontsize=25)
plt.xlabel('Datetime',fontsize=20)
plt.ylabel('Demand',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.scatter(x,y,color='green',s=20)


# In[13]:

#Observations: From the Scatter plot, its infered that the distribution would contain outliers


# In[14]:

#To visualize demand variations with respect to time on certain days(randomly)
traf_mgmt_tsd1=traf_mgmt_data[traf_mgmt_data['day']==1]
traf_mgmt_tsday1=pd.pivot_table(traf_mgmt_tsd1,index=['time'],values='demand')                                                                                              
traf_mgmt_tsd10=traf_mgmt_data[traf_mgmt_data['day']==10]
traf_mgmt_tsday10=pd.pivot_table(traf_mgmt_tsd10,index=['time'],values='demand')
traf_mgmt_tsd18=traf_mgmt_data[traf_mgmt_data['day']==18]
traf_mgmt_tsday18=pd.pivot_table(traf_mgmt_tsd18,index=['time'],values='demand')
traf_mgmt_tsd30=traf_mgmt_data[traf_mgmt_data['day']==30]
traf_mgmt_tsday30=pd.pivot_table(traf_mgmt_tsd30,index=['time'],values='demand')
traf_mgmt_tsd40=traf_mgmt_data[traf_mgmt_data['day']==40]
traf_mgmt_tsday40=pd.pivot_table(traf_mgmt_tsd40,index=['time'],values='demand')
traf_mgmt_tsd50=traf_mgmt_data[traf_mgmt_data['day']==50]
traf_mgmt_tsday50=pd.pivot_table(traf_mgmt_tsd50,index=['time'],values='demand')
traf_mgmt_tsd60=traf_mgmt_data[traf_mgmt_data['day']==60]
traf_mgmt_tsday60=pd.pivot_table(traf_mgmt_tsd60,index=['time'],values='demand')


# In[15]:

#Line Plot - Timestamp Vs Demand(mean): Selected days
plt.figure(figsize=(10,5))
plt.title('Time Vs Demand - Line plot:Daywise',fontsize=10)
plt.xlabel('Time',fontsize=10)
plt.ylabel('Demand',fontsize=10)
plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.plot(traf_mgmt_tsday1,color='red',label='day1')
plt.plot(traf_mgmt_tsday10,color='blue',label='day10')
plt.plot(traf_mgmt_tsday18,color='green',label='day18')
plt.plot(traf_mgmt_tsday30,color='yellow',label='day30')
plt.plot(traf_mgmt_tsday40,color='black',label='day40')
plt.plot(traf_mgmt_tsday50,color='orange',label='day50')
plt.plot(traf_mgmt_tsday60,color='cyan',label='day60')
plt.legend(loc='best',fontsize=10)


# In[16]:

#Observations: Above Time Vs Demand plot strongly shows that the demand is high between 5 to 12 Hrs approximately with peak at 
#9 to 11 Hrs
#Also there seems an unusual fall in demand on day18 during the peak hours - An Outlier that need to be removed 


# In[17]:

#Checking for outliers
traf_mgmt_data.describe()


# In[18]:

#Using quartile and Interquartile ranges for finding outlier
#Interquartile Range=Q3-Q1
IQR=traf_mgmt_data.describe().loc['75%','demand']-traf_mgmt_data.describe().loc['25%','demand']
print('IQR',IQR)
#Low Range=Q1-1.5*IQR
Lowrange_Outlier=traf_mgmt_data.describe().loc['25%','demand']-(1.5*IQR)
print('Low Range Outlier',Lowrange_Outlier)
#High Range=Q3+1.5*IQR
Highrange_Outlier=traf_mgmt_data.describe().loc['75%','demand']+(1.5*IQR)
print('High Range Outlier',Highrange_Outlier)


# In[19]:

#Observation: Comparing the Outliers with Minimum and Maximum values in Describe, its evident that there are outliers in 
#High Range


# In[20]:

#Removing the outlier
traf_mgmt_outlier=traf_mgmt_data['demand']>Highrange_Outlier
traf_mgmt_nooutlier=traf_mgmt_data[-traf_mgmt_outlier]


# In[21]:

#Also the unusual drop in day18 is removed
traf_mgmt_nooutlier=traf_mgmt_nooutlier[traf_mgmt_nooutlier['demand']>0.025]


# In[22]:

#Groupby after removing the outliers
traf_mgmt_grp=traf_mgmt_nooutlier.groupby(traf_mgmt_nooutlier.index).mean()
traf_mgmt_grp


# In[23]:

#Creating pandas Series using the given DataFrame
traf_mgmt_series = pd.Series(traf_mgmt_grp['demand'], index=traf_mgmt_grp.index)
traf_mgmt_series.head()


# In[24]:

#Checking Stationarity: Using Rolling mean/std and Dickey-Fuller Test
#Plotting mean and variance - To check how far they are constant over time(Stationary time series) 
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn import metrics 

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rollmean = pd.rolling_mean(timeseries, window=96)
    rollstd = pd.rolling_std(timeseries, window=96)

    #Plot rolling statistics:
    plt.figure(figsize=(30,10))
    plt.xlabel('Datetime',fontsize=20)
    plt.ylabel('Demand',fontsize=20)
    plt.xticks(rotation=90,fontsize=20)
    plt.yticks(fontsize=20)
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rollmean, color='red', label='Rolling Mean')
    std = plt.plot(rollstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best',fontsize=15)
    plt.title('Rolling Mean & Standard Deviation',fontsize=20)
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        
    print (dfoutput)

#Calling the function that checks stationarity
test_stationarity(traf_mgmt_series)


# In[25]:

#Observation: No Increasing/Decreasing trend in mean
#Variation in Standard Deviation is low
#Test Static is lesser than Critical Values and is more negative - Trend Stationary
#p-value is less than 0.05 - Trend Stationary
#Variation in mean need to be reduced


# In[27]:

#Log transformation of data

traf_mgmt_serieslog = np.log(traf_mgmt_series)
plt.figure(figsize=(30,10))
plt.xlabel('Datetime',fontsize=20)
plt.ylabel('Demand',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.plot(traf_mgmt_serieslog )


# In[28]:

#Eliminating noise by Smoothing - taking rolling averages
traf_mgmt_seriesmovavg = pd.rolling_mean(traf_mgmt_serieslog,96)
plt.figure(figsize=(30,10))
plt.xlabel('Datetime',fontsize=20)
plt.ylabel('Demand',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.plot(traf_mgmt_serieslog)
plt.plot(traf_mgmt_seriesmovavg, color='red')


# In[30]:

#Smoothing to reduce variation in mean
traf_mgmt_seriesmovavg_diff=traf_mgmt_serieslog-traf_mgmt_seriesmovavg 
traf_mgmt_seriesmovavg_diff.dropna(inplace=True)

#Calling the function that checks stationarity after smoothing
test_stationarity(traf_mgmt_seriesmovavg_diff)


# In[48]:

#Observation: This looks like a much better series. The rolling values appear to be varying slightly and seems smoothened


# In[31]:

#Eliminating Seasonality by differencing the observation at a particular instant and that of the previous instant
traf_mgmt_serieslogdiff=traf_mgmt_serieslog - traf_mgmt_serieslog.shift()
plt.figure(figsize=(30,10))
plt.xlabel('Datetime',fontsize=20)
plt.ylabel('Demand',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.plot(traf_mgmt_serieslogdiff)
traf_mgmt_serieslogdiff.dropna(inplace=True)

#Calling the function that checks stationarity after differencing
test_stationarity(traf_mgmt_serieslogdiff)


# In[120]:

#We can see that the mean and std variations have small variations with time. No Seasonality being observed.
#thus the TS is stationary 


# In[32]:

#Plots to determine the parameters of the ARIMA model(p,d,q)
#ACF and PACF plots after differencing 
from statsmodels.tsa.stattools import acf, pacf
traf_mgmt_acf = acf(traf_mgmt_serieslogdiff, nlags=10)
traf_mgmt_pacf = pacf(traf_mgmt_serieslogdiff, nlags=10, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(traf_mgmt_acf )
plt.axhline(y=0,linestyle='--',color='orange')
plt.axhline(y=-1.96/np.sqrt(len(traf_mgmt_serieslogdiff)),linestyle='--',color='yellow')
plt.axhline(y=1.96/np.sqrt(len(traf_mgmt_serieslogdiff)),linestyle='--',color='green')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(traf_mgmt_pacf )
plt.axhline(y=0,linestyle='--',color='orange')
plt.axhline(y=-1.96/np.sqrt(len(traf_mgmt_serieslogdiff)),linestyle='--',color='yellow')
plt.axhline(y=1.96/np.sqrt(len(traf_mgmt_serieslogdiff)),linestyle='--',color='green')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
#p = q= 1 and d =1(diff order of 1)


# In[33]:

#ARIMA model
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(traf_mgmt_serieslog , order=(1, 1, 1))  
traf_mgmt_ARIMA = model.fit(disp=-1)  
plt.plot(traf_mgmt_serieslogdiff)
plt.plot(traf_mgmt_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((traf_mgmt_ARIMA.fittedvalues-traf_mgmt_serieslogdiff)**2))


# In[34]:

#Adding noise and seasonality to the predicted residuals 
#Scaling it back to the original values  


# In[35]:

#Predicated Residuals
traf_mgmt_ARIMA_diff = pd.Series(traf_mgmt_ARIMA.fittedvalues, copy=True)
traf_mgmt_ARIMA_diff.head()


# In[36]:

#cumsum of predicted values
traf_mgmt_ARIMA_diff_cumsum = traf_mgmt_ARIMA_diff.cumsum()
traf_mgmt_ARIMA_diff_cumsum.head()


# In[37]:

#To introduce back the seasonality

traf_mgmt_ARIMA_log=pd.Series(traf_mgmt_serieslog.ix[0], index=traf_mgmt_serieslog.index)
traf_mgmt_ARIMA_log = traf_mgmt_ARIMA_log.add(traf_mgmt_ARIMA_diff_cumsum,fill_value=0)
traf_mgmt_ARIMA_log.head()


# In[65]:

#Check with originals
traf_mgmt_ARIMA_pred=np.exp(traf_mgmt_ARIMA_log)
plt.figure(figsize=(20,10))
plt.plot(traf_mgmt_tseries)
plt.plot(traf_mgmt_ARIMA_pred)
plt.title('RMSE: %.4f'% np.sqrt(sum((traf_mgmt_ARIMA_pred-traf_mgmt_tseries)**2)/len(traf_mgmt_tseries)))


# In[40]:

#Outputting the model

def ARIMA_mod(traf_mgmt_series):
    model = ARIMA(traf_mgmt_series, order=(1,1,1))  
    traf_mgmt_ARIMAmodtst = model.fit(disp=-1)
    #Getting the residuals
    traf_mgmt_ARIMA_resi= pd.Series(traf_mgmt_ARIMAmodtst.fittedvalues, copy=True)
    #cumsum of residuals
    traf_mgmt_ARIMA_resi_cumsum= traf_mgmt_ARIMA_resi.cumsum()
    #adding residuals to original
    traf_mgmt_series_logtst=pd.Series(traf_mgmt_series.ix[0], index=traf_mgmt_series.index)
    traf_mgmt_ARIMA_logtst = traf_mgmt_series_logtst.add(traf_mgmt_ARIMA_resi_cumsum,fill_value=0)
    #Reversing Log
    traf_mgmt_ARIMA_predtst=np.exp(traf_mgmt_ARIMA_logtst)
    print(traf_mgmt_ARIMA_predtst.head(30))
    plt.figure(figsize=(10,10))
    plt.plot(traf_mgmt_testseries)
    plt.plot(traf_mgmt_ARIMA_predtst)
    plt.title('RMSE: %.4f'% np.sqrt(sum((traf_mgmt_ARIMA_predtst-traf_mgmt_series)**2)/len(traf_mgmt_series)))

#Calling ARIMA model using the Test Series
traf_mgmt_datatest=traf_mgmt_data[traf_mgmt_data['day']>40]
traf_mgmt_testgrp=traf_mgmt_datatest.groupby(traf_mgmt_datatest.index).mean()
traf_mgmt_testseries=pd.Series(traf_mgmt_testgrp['demand'], index=traf_mgmt_testgrp.index)

traf_mgmt_testserieslog = np.log(traf_mgmt_testseries)

ARIMA_mod(traf_mgmt_testserieslog)



# In[ ]:




# In[ ]:




# In[ ]:



