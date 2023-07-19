import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix = []
    for i in range(len(vectorSeries) - sequence_length + 1):
        matrix.append(vectorSeries[i:i + sequence_length])
    return matrix


# random seed
np.random.seed(1234)

# load the data
path_to_dataset = 'result/vix1.csv'
sequence_length = 30

# vector to store the time series
vector_vix = []

# read the file using pandas and remove empty rows
df = pd.read_csv(path_to_dataset)
df = df.dropna()

# extract the relevant column from the DataFrame
vector_vix = df['vix'].values.tolist()  # replace 'column_name' with the actual column name from the dataset

# convert the vector series to a matrix
matrix_vix = convertSeriesToMatrix(vector_vix, sequence_length)

# shift all data by mean
matrix_vix = np.array(matrix_vix)
shifted_value = matrix_vix.mean()
matrix_vix -= shifted_value
print("Data shape:", matrix_vix.shape)

# split dataset: training set and test set
train_set = matrix_vix[:1433, :]
test_set = matrix_vix[1433:, :]

# shuffle the training set
np.random.shuffle(train_set)

# the training set
X_train = train_set[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1]
# the test set
X_test = test_set[:, :-1]
y_test = test_set[:, -1]

# the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建模型
model = Sequential()
# 第一层 LSTM
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))
# 第二层 LSTM
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
# 第三层 Dense
model.add(Dense(units=1, activation='linear'))
# 编译模型

model.compile(loss="mse", optimizer='rmsprop')

# 训练模型
history = model.fit(X_train, y_train, batch_size=60, epochs=300, validation_split=0.05, verbose=1)

# evaluate the result
train_mse = model.evaluate(X_train, y_train, verbose=1)
test_mse = model.evaluate(X_test, y_test, verbose=1)
print('The mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test)))
print('The mean squared error (MSE) on the train data set is %.3f over %d test samples.' % (train_mse, len(y_train)))
# plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()
# get the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples, 1))

# get the dates for the test set
test_dates = df['Date'].values[1433 + sequence_length - 1:]

# convert test_dates to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

# plot the results
fig, ax = plt.subplots()
ax.plot(test_dates, y_test + shifted_value)
ax.plot(test_dates, predicted_values + shifted_value)
ax.set_xlabel('Date')
ax.set_ylabel('VIX')

# set x-axis tick locator and formatter
locator = mdates.MonthLocator()  # set locator to display ticks at monthly intervals
formatter = mdates.DateFormatter('%Y-%m-%d')  # set formatter to display dates in 'YYYY-MM-DD' format
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.legend(['Training', 'Validation'])
plt.xticks(rotation=45)  # rotate x-axis labels for better visibility
plt.show()
