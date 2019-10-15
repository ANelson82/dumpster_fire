import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

data_set = pd.read_csv('international-airline-passengers.csv')
data_set['Month']=pd.to_datetime(data_set['Month'])
data_set = data_set.set_index('Month')

decomposition = sm.tsa.seasonal_decompose(data_set, model = 'additive')

fig, ax = plt.subplots()
ax.grid(True)
year = mdates.YearLocator(month=1)
month = mdates.MonthLocator(interval=3)
year_format = mdates.DateFormatter('%Y')
month_format = mdates.DateFormatter('%m')
ax.xaxis.set_minor_locator(month)
ax.xaxis.grid(True, which = 'minor')
ax.xaxis.set_major_locator(year)
ax.xaxis.set_major_formatter(year_format)
plt.plot(data_set.index, data_set['Passengers'], c='blue')
plt.plot(decomposition.trend.index, decomposition.trend, c='red')
pd.plotting.register_matplotlib_converters() #futurewarning error
plt.show()

#https://towardsdatascience.com/analyzing-time-series-data-in-pandas-be3887fdd621
#https://www.kaggle.com/andreazzini/international-airline-passengers/download
