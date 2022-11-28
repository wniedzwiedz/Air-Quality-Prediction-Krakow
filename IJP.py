import pandas as pd

def getIndex(PM10=0,PM25=0,NO2=0,SO2=0,O3=0):
    if (PM10<=20 and PM25<=13 and NO2<=40 and SO2<=50 and O3<=70):
        return 1
    elif (PM10<=50 and PM25<=35 and NO2<=100 and SO2<=100 and O3<=120):
        return 2
    elif (PM10<=80 and PM25<=55 and NO2<=150 and SO2<=200 and O3<=150):
        return 3
    elif (PM10<=110 and PM25<=75 and NO2<=200 and SO2<=350 and O3<=180):
        return 4
    elif (PM10<=150 and PM25<=110 and NO2<=400 and SO2<=500 and O3<=240):
        return 5
    else:
        return 6
    
    
def plotIndex(PM10,PM25,NO2,SO2,O3):
    li = []
    #li.append(PM10.index)
    li.append(PM10)
    li.append(PM25)
    li.append(NO2)
    li.append(SO2)
    li.append(O3)
    
    levels=["Bardzo dobry","Dobry","Umiarkowany","Dostateczny","Zły","Bardzo zły"]

    df = []
    cat=[]
    #df.set_index('Czas')
    
    d=pd.concat(li, axis=1, ignore_index=False)#.set_index('Czas')
    
    for row in d.iterrows():
        df.append(getIndex(row[1]['PM10'],row[1]['PM25'],row[1]['NO2'],row[1]['SO2'],row[1]['O3']))
        cat.append(levels[getIndex(row[1]['PM10'],row[1]['PM25'],row[1]['NO2'],row[1]['SO2'],row[1]['O3'])-1])
    
    return d.index,df,cat