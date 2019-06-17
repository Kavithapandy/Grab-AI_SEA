
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


# In[19]:

#Reading input file
traf_mgmt_file=your_local_path+'training.csv'
print(traf_mgmt_file)
traf_mgmt_data=pd.read_csv(traf_mgmt_file)
#Head of input
traf_mgmt_data.head()


# In[20]:

#Converting Geohash to its respective Latitude and Longitude
import geohash2 as pgh
traf_mgmt_data['latlong']=traf_mgmt_data['geohash6'].apply(lambda x:pgh.decode(x))
traf_mgmt_data.tail()


# In[21]:

#Splitting the latitude and longitude
traf_mgmt_data[['lat','long']]=pd.DataFrame(traf_mgmt_data['latlong'].tolist(),index=traf_mgmt_data.index)
traf_mgmt_data.tail()


# In[24]:

#Identifying the unique latlong, geohash6 points
unique_geohash6=traf_mgmt_data['geohash6'].unique().tolist()
unique_latlong=traf_mgmt_data['latlong'].unique().tolist()
print('latlong:', len(traf_mgmt_data['latlong'].unique()))
print('unique geohash6:', len(traf_mgmt_data['geohash6'].unique()))


# In[26]:

#converting the coordinates as radians
rad=lambda x:np.radians(float(x))
traf_mgmt_data['radlat']=traf_mgmt_data['lat'].apply(rad)
traf_mgmt_data['radlong']=traf_mgmt_data['long'].apply(rad)


# In[27]:

#traf_mgmt_data=traf_mgmt_data.set_index(['latlong'])
traf_mgmt_data.head()


# In[34]:

traf_mgmt_datagrp=traf_mgmt_data.groupby(traf_mgmt_data.index).mean()
traf_mgmt_datagrp


# In[37]:

traf_mgmt_datagrp.describe()


# In[ ]:

#From Describe, max(demand)=0.388203
#min(demand)=0.004986
#From the histogram, only 4 Lat/Long points have high demand, >0.25


# In[43]:

#Plotting the demands grouped by Lat/Long
plt.figure(figsize=(30,10))
plt.title('Demand - Grouped by Lat/Long',fontsize=25)
plt.xlabel('Demand',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.hist(traf_mgmt_datagrp['demand'],color='green')


# In[49]:

#Getting the Latitude /Longitude points of high demand
#categorizing demand>0.2 as high
#Only few area are with high demand and are below
high_demand_space=traf_mgmt_datagrp[traf_mgmt_datagrp['demand']>0.2]
high_demand_space


# In[ ]:



