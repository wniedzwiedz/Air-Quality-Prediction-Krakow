from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import warnings 


def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def predict(data):
    train, test = ARIMA.split(data)
    fit1 = SimpleExpSmoothing(train).fit(optimized=True)
    #print(fit1.summary())
    fcast1 = fit1.forecast(len(test)).rename("Proste wygłądzanie wykładnicze")
    test = test.rename('Pomiary')
    plt.figure()
    test.plot(legend=True)
    fcast1.plot(legend=True)
    
    rmse = MAPE(test, fcast1)
    print('MAPE: %.3f'% rmse)
    
def predict1(data):
    train, test = ARIMA.split(data)
    
    fit1 = Holt(train).fit()
    fcast1 = fit1.forecast(len(test)).rename("Trend addytywny")
    fit2 = Holt(train, exponential=True).fit()
    fcast2 = fit2.forecast(len(test)).rename("Trend multiplikatywny")
    fit3 = Holt(train, damped_trend=True).fit()
    fcast3 = fit3.forecast(len(test)).rename("Trend addytywny tłumiony")
    fit4 = Holt(train, damped_trend=True, exponential=True).fit()
    fcast4 = fit4.forecast(len(test)).rename("Trend multiplikatywny tłumiony")

    
    test = test.rename('Pomiary')
    plt.figure()
    
    test.plot(legend=True)
    
    fcast1.plot(legend=True)
    fcast2.plot(legend=True)
    fcast3.plot(legend=True)
    fcast4.plot(legend=True)
    
    rmse = MAPE(test, fcast1)
    print('MAPE: %.3f'% rmse)
    rmse = MAPE(test, fcast2)
    print('MAPE: %.3f'% rmse)
    rmse = MAPE(test, fcast3)
    print('MAPE: %.3f'% rmse)
    rmse = MAPE(test, fcast4)
    print('MAPE: %.3f'% rmse)
    
    return fcast2.rename("Prognoza")
    
def predict2(data):
    warnings.filterwarnings("ignore") 
    train, test = ARIMA.split(data)
    fit1 = ExponentialSmoothing(train, seasonal_periods=52, trend='add', seasonal='add').fit()
    fit2 = ExponentialSmoothing(train, seasonal_periods=52, trend='mul', seasonal='mul').fit()
    fit3 = ExponentialSmoothing(train, seasonal_periods=52, trend='add', seasonal='mul', damped_trend=True).fit()
    fit4 = ExponentialSmoothing(train, seasonal_periods=52, trend='mul', seasonal='mul', damped_trend=True).fit()
    
    #print(fit1.summary())
  #  fit1.fittedvalues.plot(style='--', color='red')
   # fit2.fittedvalues.plot(style='--', color='green')
    test = test.rename('Pomiary')
    plt.figure()
    test.plot(legend=True)
    
    fit1.forecast(len(test)).rename("Sezonowość addytywna").plot(legend=True)
    fit2.forecast(len(test)).rename("mulmul").plot(legend=True)
    fit3.forecast(len(test)).rename("Sezonowość multiplikatywna").plot(legend=True)
    fit4.forecast(len(test)).rename("mulmuldamp").plot(legend=True)
    
    
    rmse = MAPE(test, fit1.forecast(len(test)))
    print('MAPE: %.3f'% rmse)
    #rmse = sqrt(mean_squared_error(test, fit2.forecast(len(test))))
    #print('RMSE: %.3f'% rmse)
    rmse = MAPE(test, fit3.forecast(len(test)))
    print('MAPE: %.3f'% rmse)
    #rmse = sqrt(mean_squared_error(test, fit4.forecast(len(test))))
    #print('RMSE: %.3f'% rmse)
    return fit1.forecast(len(test)).rename("Prognoza") 