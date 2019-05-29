import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
# import quandl
# dataset =  pd.read_csv("https://www.quandl.com/api/v3/datasets/EOD/MSFT.csv?qopts.export=true&api_key=7yqZ9genR5nx7BAuMzqj")
# print(dataset.head())
dataset = pd.read_csv("EOD-MSFT_weekly.csv")
dataset['Date'] = pd.to_datetime(dataset.Date , format='%Y-%m-%d')
dataset.index = dataset['Date']
dataset = dataset.reindex(index = dataset.index[::-1])
df = dataset['Close']

#Plotting dataset
# print(df.head())
# print(df.tail())
# print(df.shape)
plt.xlabel("Date")
plt.ylabel("Stock Market Price of MS")
plt.plot(df)
plt.title("Stock Market Daily Variations")
plt.show()

def test_stationarity(dataframe , title):
    #Rolling statistics
    rol_mean = dataframe.rolling(window = 12).mean()
    rol_std = dataframe.rolling(window = 12).std()
    rol_mean = rol_mean.dropna()
    rol_std = rol_std.dropna()
    print(rol_mean.head(2))
    print(rol_mean.tail(2))
    print()
    # print(rol_std.head(3))
    # print(rol_std.tail(3))

    #Visualizing Rolling statistics
    plt.plot(dataframe , color='blue' , label = 'Original')
    plt.plot(rol_mean , color = 'red' , label = 'Rolling Mean')
    plt.plot(rol_std , color = 'black' , label = 'Rolling Standard Deviation')
    plt.legend(loc = 'best')
    plt.title(title)
    plt.show()

    # Augmented Dickey-Fuller Test
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(dataframe)
    test_statistics = result[0]
    print()
    print('ADF/TEST statistics = %f' %test_statistics)
    print('P-value = %f' %result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.5f' % (key, value))
        if test_statistics > value:
            print("\t\t NOT Stationary as ADF/TEST statistics is greater than %s Critical value" %key)
        else:
            print('\t\t Stationary')

#Checking Stationarity
print()
print('Stationarity TEST of df')
print()
test_stationarity(df , "Original Dataframe")
log_df = np.log(df)
log_df_rol_mean = log_df.rolling(window=12).mean()
log_df_weighted_mean = log_df.ewm(halflife = 12 , min_periods = 0 , adjust = True).mean()
plt.plot(log_df , label = 'log df')
plt.plot(log_df_rol_mean , color ='red' , label = 'log_df_rol_mean')
plt.plot(log_df_weighted_mean , color = 'green' , label = 'log_df_weighted_mean')
plt.title('Log Of dataset and its mean and weighted mean')
plt.legend(loc='best')
plt.show()
print()
print()
print()
print('Stationarity TEST of log_df_rol_mean')
print()
new_df = log_df - log_df_rol_mean
new_df = new_df.dropna()
test_stationarity(new_df , "Log Data - Rolling mean")
new_df1 = log_df - log_df_weighted_mean
new_df1 = new_df1.dropna()
print()
print()
print()
print('Stationarity TEST of log_df_weighted_mean')
print()
test_stationarity(new_df1 , "Log Data - Weighted mean")
new_df2 = log_df - log_df.shift() # same as df.diff()
new_df2 = new_df2.dropna()
print()
print()
print()
print('Stationarity TEST of differentiated df')
print()
test_stationarity(new_df2 , 'Stationarity TEST of differentiated df')

#Visualizing trend, seasonal and resdiual errors(noise)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df , freq=12)

trend = decomposition.trend
seasonal =decomposition.seasonal
residual = decomposition.resid

plt.subplot(4 , 1, 1)
plt.plot(log_df , color = 'blue' , label = 'Original')
plt.legend(loc = 'best')
plt.subplot(4 , 1, 2)
plt.plot(trend , color = 'black' , label = 'Trend')
plt.legend(loc = 'best')
plt.subplot(4 , 1, 3)
plt.plot(seasonal , color = 'green' , label = 'Seasonality')
plt.legend(loc = 'best')
plt.subplot(4 , 1, 4)
plt.plot(residual , color = 'red' , label = 'Residual Errors(Noise)')
plt.legend(loc = 'best')
plt.show()

# residual = residual.dropna()
# test_stationarity(residual , "Noise")

def arima_fit(dataframe , title ,p , q):
    #Checking for acf and pacf plots for the values of p and q
    from statsmodels.tsa.stattools import acf , pacf
    diff_acf = acf(dataframe , nlags=40)
    diff_pacf = pacf(dataframe , nlags=40 , method='ywunbiased')

    plt.subplot(1 , 2 , 1)
    plt.plot(diff_acf , color = 'red' , label= 'Autocorrelation Function')
    plt.xlim(0 ,10)
    plt.xticks(ticks=np.arange(1,20,1))
    plt.legend(loc= 'best')
    plt.subplot( 1,2, 2)
    plt.plot(diff_pacf , color = 'black' , label = 'Partial Autocorrelation Function')
    plt.xlim(0 ,10)
    plt.xticks(ticks=np.arange(1,20,1))
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

    #Fitting the Arima Model
    from statsmodels.tsa.arima_model import ARIMA
    dataframe = dataframe.dropna()
    print(dataframe.head(10))
    model = ARIMA(dataframe , order=(p,1,q))
    results_ARIMA = model.fit(disp=-1)
    plt.plot(log_df , color = 'black')
    plt.plot(results_ARIMA.fittedvalues , color='red')
    plt.title("Arima Fitted Values %s" %title)
    plt.show()
    # print(model.summary())

    predictions_Arima_diff = pd.Series(results_ARIMA.fittedvalues , copy=True)
    print(predictions_Arima_diff.head())

    predictions_Arima_diff_cumsum = predictions_Arima_diff.cumsum()
    print(predictions_Arima_diff_cumsum.head(10))

    predictions_Arima_log = pd.Series(log_df.iloc[1] , index=log_df.index)
    predictions_Arima_log = predictions_Arima_log.add(predictions_Arima_diff_cumsum , fill_value= 0)
    print(predictions_Arima_log.head(10))

    predictions_ARIMA = np.exp(predictions_Arima_log)
    # print(predictions_ARIMA)
    # print(df)
    plt.xlabel("Date")
    plt.ylabel("Stock Market Price of MS")
    plt.plot(df , label = "Original")
    plt.plot(predictions_ARIMA , color='red' , label = 'Predicted %s' %title)
    plt.title("Predictions vs Original")
    plt.legend(loc = 'best')
    plt.show()

    results_ARIMA.plot_predict('2014' , '2019')
    plt.show()

arima_fit(new_df , "new_df" , 8 , 2)
arima_fit(new_df1 , "new_df1" , 8 , 3)
arima_fit(new_df2 , 'new_df2', 4, 4)
