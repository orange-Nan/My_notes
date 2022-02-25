#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans


f = r'C:\Users\LULU\Desktop\biyelunwen\shuju\Shenyang_meteorological_data.csv'
sheet = pd.read_csv(f,header=0)
data = DataFrame(sheet)

#去除异常值-999
data = data.replace(-9999,np.nan)
data = data.dropna()

data['date'] = pd.to_datetime(data['date'],format = '%Y/%m/%d %H:00')
data['T'] = data['T'].map(lambda x:x/10)

print(len(data['T']))  #15839


y1 = list(data['T'])
y2 = np.array(y1)
y2 = y2.reshape(-1,1)


#Kmeans
y = KMeans(n_clusters=50).fit_predict(y2)

fig = plt.figure(figsize=(10,10),dpi=200)
ax1=fig.add_subplot(111)
ax1.scatter(data['date'],data['T'],c=y)
ax1.set_xlabel('Time',fontsize=15)
ax1.set_ylabel('Temperature(°C)',fontsize=15)
plt.suptitle('Distribution of Temperature in Shenyang',fontsize=20,y=0.92)
plt.savefig('C:/Users/LULU/Desktop/Kmean.jpg')
plt.show()


# In[ ]:




