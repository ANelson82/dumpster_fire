import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pyplot import figure
from datetime import datetime
import statsmodels.api as sm

data_set = pd.read_csv('international-airline-passengers.csv')
data_set['Month']=pd.to_datetime(data_set['Month'])
data_set = data_set.set_index('Month')

#start_date = datetime(1959,1,1)
#end_date = datetime(1960,12,1)
#data_set[(start_date <=data_set.index) & (data_set.index <= end_date)].plot(grid=True)
#data_set.plot(grid=True)


decomposition = sm.tsa.seasonal_decompose(data_set, model = 'additive')
#figure(figsize = (12, 8)
fig = decomposition.plot()
plt.show()
#print(data_set.dtypes)
#print(data_set.head(3))
