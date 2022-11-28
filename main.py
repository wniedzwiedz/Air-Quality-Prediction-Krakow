#%%
import read
import matplotlib.pyplot as plt
import pandas as pd
import ARIMA
import pickle

import IJP
import glob
from pmdarima.arima import ADFTest

#%%
O3 = pd.read_pickle("O3.pkl")["MpKrakBujaka"]
PM10 = pd.read_pickle("PM10.pkl")["MpKrakBujaka"]
PM25 = pd.read_pickle("PM25.pkl")["MpKrakBujaka"]
SO2 = pd.read_pickle("SO2.pkl")["MpKrakBujaka"]
NO2 = pd.read_pickle("NO2.pkl")["MpKrakBujaka"]

#%%
def getdata(f):
    f=f[(f.index >= '2014-1-1 00:00:00')]
    f=f.resample('W').max().interpolate(limit_area="inside").dropna().astype(int)
    return f

PM10=getdata(PM10)
PM25=getdata(PM25)
SO2=getdata(SO2)
NO2=getdata(NO2)
O3=getdata(O3)


#%%
k=PM10
train, test = ARIMA.split(k)
#ARIMA.fitModel(k)         

#%%
j=ARIMA.difference(train,1)
print(j)

plt.figure()
plt.plot(j.index,j)
plt.draw()

#%%
import ARIMA
ARIMA.plots(train,104)
j=ARIMA.difference(train,52)
ARIMA.plots(j,103)
#j=ARIMA.difference(j,52)
#ARIMA.plots(j,103)

#%%
import ARIMA
data=PM10
h=ARIMA.predict2(data)

#%%
rem=(data-h).rename("Różnica międzi pomiarem a prognozą") 
rem=rem[(rem.index >= '2019-1-1 00:00:00')]
plt.figure()
rem.plot()
ARIMA.plots(rem,10)
#%%
import ES
#ES.predict(data)

for data in [PM10,PM25,NO2,SO2,O3]:
    ES.predict(data)
    ES.predict1(data)
    ES.predict2(data)

#%%
import IJP

def get(name):
    d = pd.read_pickle(name+".pkl")
    train, res = ARIMA.split(d)
    
    res=res.resample('W').max()
    res=res[(res.index >= '2014-1-1 00:00:00')]
    res=res["MpKrakBujaka"].interpolate(limit_area="inside").dropna().astype(int)
    
    
    return res.rename(name)

PM10=get("PM10")
PM25=get("PM25")
NO2=get("NO2")
SO2=get("SO2")
O3=get("O3")

levels=["Dobry","Umiarkowany","Dostateczny","Zły","Bardzo zły"]

#%%
r=IJP.plotIndex(PM10, PM25, NO2, SO2, O3)
plt.figure()
plt.plot(r[0][0:5],levels,'w,')
plt.plot(r[0],r[2],'-o')
plt.grid(axis='y')
plt.title("Indeks jakości powietrza")



#%%
import IJP

O3_p = pd.read_pickle("O3_p.pkl")
PM10_p = pd.read_pickle("PM10_p.pkl")
PM25_p = pd.read_pickle("PM25_p.pkl")
SO2_p = pd.read_pickle("SO2_p.pkl")
NO2_p = pd.read_pickle("NO2_p.pkl")

rp=IJP.plotIndex(PM10_p.rename("PM10"), PM25_p.rename("PM25"),  NO2_p.rename("NO2"), SO2_p.rename("SO2"), O3_p.rename("O3"))

plt.figure()
plt.plot(rp[0][0:5],levels,'w,')
#plt.plot(r[0],r[2],'-o')
plt.plot(rp[0],rp[2],'-o')
plt.grid(axis='y')
plt.title("Indeks jakości powietrza")

#%%
diff=[]
for i in range(0,53):
    diff.append(abs(r[1][i]-rp[1][i]))
 
plt.plot(rp[0],diff,'-o')

#%%
from sklearn.metrics import confusion_matrix
import seaborn as sns

cf_matrix = confusion_matrix(r[1], rp[1])
print(cf_matrix)
df_cm = pd.DataFrame(cf_matrix, index = levels,
                  columns = levels)
sns.heatmap(df_cm, annot=True)