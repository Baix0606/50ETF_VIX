import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mydata = pd.read_csv('result\\vix1.csv', index_col=0)
mydata['Date'] = pd.to_datetime(mydata['Date'])
stdata = pd.read_csv('iVIX\\ivixx.csv')
stdata['date'] = pd.to_datetime(stdata['date'])
plt.figure(figsize=(10,3))
plt.plot(mydata['Date'], mydata['vix'])
plt.plot(stdata['date'], stdata['ivix'])
plt.legend(['CNVIX', 'iVIX'])
plt.show()

# Subset the data for the relevant period
start_date = pd.to_datetime('2015-03-02')
end_date = pd.to_datetime('2018-02-13')
mydata_subset = mydata[(mydata['Date'] >= start_date) & (mydata['Date'] <= end_date)]
stdata_subset = stdata[(stdata['date'] >= start_date) & (stdata['date'] <= end_date)]

# Merge the two datasets based on the date column
merged_data = pd.merge(mydata_subset, stdata_subset, left_on='Date', right_on='date')

# Calculate the absolute difference between VIX values
merged_data['abs_diff'] = np.abs(merged_data['vix'] - merged_data['ivix'])

# Calculate the MAE for each day
merged_data['MAE'] = merged_data['abs_diff']

# Calculate the average MAE
average_mae = merged_data['MAE'].mean()

print("Average MAE: ", average_mae)

correlation = mydata_subset['vix'].corr(stdata_subset['ivix'])
print("相关系数: ", correlation)

