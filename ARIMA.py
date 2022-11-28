import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima 
from pmdarima.arima import ADFTest
import warnings 
from statsmodels.tsa.seasonal import seasonal_decompose 
import numpy as np


def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#%%
def difference(dataset,period):
    diff = list()
    for i in range(period, len(dataset)):
        value = dataset[i] - dataset[i - period]
        diff.append(value)
    #diff.append(dataset[len(dataset)-1])
    result = adfuller(diff)
    print('ADF Statistic: %f'% result[0])
    print('p-value: %f'% result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f'% (key, value))
    return pd.Series(diff)

#%%

def plots(series,l):
    #p, q, P, Q = find_acf_pacf(series, 52)
    series=series.interpolate(limit_area="inside").dropna().astype(int)
    test = ADFTest(alpha=0.05)
    print(test.should_diff(series))
    
    plot_acf(series, lags=l, fft=True,zero=False).show()
    plot_pacf(series, lags=l,zero=False).show()
   # print(p, q, P, Q)
    
#%%

def split(d):
    #train_size = int(len(data) -30)
    train, test = d[(d.index < '2019-1-1 00:00:00')], d[(d.index >= '2019-1-1 00:00:00')]
    return train, test
#%%

def predict(data):
    train, test = split(data)
    history = [x for x in train]
    print(train)
    predictions = list()
    for i in range(0,len(test)):
        model = ARIMA(history, order=(1,1,2))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        obs = test[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%.3f'% (yhat, obs))
    rmse = sqrt(mean_squared_error(test, predictions))
    print('RMSE: %.3f'% rmse)
    return test,predictions
        
def predict2(data):
    train, test = split(data)
    #history = [x for x in train]
    #print(test)
    
    #model = ARIMA(train, order=(1,1,1),seasonal_order=(0,1,1,52))
    model = SARIMAX(train,order=(0, 0, 1),seasonal_order=(1,1,1,52))

    model_fit = model.fit()
    #model_fit.summary()
    
    
    
    start = len(train) 
    end = len(train) + len(test) - 1
      
    # Predictions for one-year against the test set 
    predictions = model_fit.predict(start, end, 
                                 typ = 'levels').rename("Prognoza") 
      
    # plot predictions and actual values
    predictions.plot(legend = True) 
    
    test = test.rename('Pomiary')
    test.plot(legend = True) 

    
        
    #predictions=model_fit.forecast(steps=len(test))
    rmse = MAPE(test, predictions)
    print('MAPE: %.3f'% rmse)
    return predictions
    
def fitModel(data):
  
    # Ignore harmless warnings 
    warnings.filterwarnings("ignore") 
    
    test = ADFTest(alpha=0.05)
    print(test.should_diff(data))
    
    seasonal_decompose(data).plot() 
    
    # Fit auto_arima function 
    stepwise_fit = auto_arima(data.astype(int), start_p = 3, start_q = 5, 
                              start_Q = 1, start_P = 2,
                              D=1, m = 52, max_p=3,
                              seasonal = True, 
                               trace = True, 
                              error_action ='ignore',   # we don't want to know if an order does not work 
                              suppress_warnings = True,  # we don't want convergence warnings 
                              stepwise = True, approximate=False)           # set to stepwise 
      
    # To print the summary 
    stepwise_fit.summary() 
    
        