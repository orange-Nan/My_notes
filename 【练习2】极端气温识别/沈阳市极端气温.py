import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans

#导入数据
f = r'C:\Users\LULU\Desktop\Shenyang_meteorological_data.csv'
sheet = pd.read_csv(f,header=0)
data = DataFrame(sheet)

#去除异常值-9999
data = data.replace(-9999,np.nan)
data = data.dropna()

data['date'] = pd.to_datetime(data['date'],format = '%Y/%m/%d %H:00')   #转换为datetime格式
data['T'] = data['T'].map(lambda x:x/10)   #原始数据是乘了十倍的
#print(len(data['T']))  #一共15839行数据

#转换为符合Kmeans要求的数组
y1 = list(data['T'])
y2 = np.array(y1)
y2 = y2.reshape(-1,1)

#Kmeans识别极端气温（另一个是识别PM2.5污染事件）
y = KMeans(n_clusters=15).fit_predict(y2)

#画图
fig = plt.figure(figsize=(10,10),dpi=200)
ax1=fig.add_subplot(111)
ax1.scatter(data['date'],data['T'],c=y)  #使用date和T画散点图，Kmeans分成的不同的类别用不同颜色标出
ax1.set_xlabel('Time',fontsize=15)
ax1.set_ylabel('Temperature(°C)',fontsize=15)
plt.suptitle('Distribution of Temperature in Shenyang',fontsize=20,y=0.92)
plt.savefig('C:/Users/LULU/Desktop/Kmean_Temperature.jpg')
plt.show()
