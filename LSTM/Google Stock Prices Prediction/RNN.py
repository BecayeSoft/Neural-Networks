# prices before each January day

# don't concatenate train+test => leak & change test values

# get 60 previous inputs before predicting


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Importing the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

print('dataset_train', dataset_train.head(2))
print(dataset_train.tail(2))

print('training_set', training_set[0:2])
print(dataset_train.info())


# Feature Scaling
"""
With RNNs, a Sigmoid output function, it's recommended (by Hadelin) to use Normalization.
"""
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

print('training_set_scaled', training_set_scaled[0:5])
print('training_set_scaled.shape', training_set_scaled.shape)


# Creating a data structure with 60 timesteps and 1 output
"""
60 steps => 60 financial days => 3 months
Use 3 months to predict the price

i = 60
previous_prices = X_train = [60-60:60, 0] = [0:60, 0]      # from 0 to 59 with 
target_price = y_train = 60

i = 61
previous_prices = X_train = [61-60:61, 0] = [1:60, 0]
target_price = y_train = 61

...

i = 1257
previous_prices = X_train = [1257-60:1257, 0] = [1197:1257, 0]
target_price = y_train = 1257
"""
X_train = []
y_train = []
for i in range(60, 1258):
    # 0 => take the 1st element in the array of 60 prices
    # because there is only one column and we want to
    # extract a 1D array
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
"""
TODO: Undertsnad RNNs input shape
"""
print('X_train.shape', X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
print('X_train.reshaped', X_train.shape)


# ----------------------------------------
# Part 2 - Building and Training the RNN

# Initialising the RNN
regressor = Sequential()

# Adding the LSTM layers
""""
50 neurons - input_shape correspond to the price (second element) and the indicator (last element)
20% = 10 neurons will be dropped out
NB: We don't return any sequence in the penultimate layer (the 4th)
"""
# regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

# Compiling the RNN
"""
Mean_squared_error since we are predicting prices (Regression).
"""
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# ----------------------------------------------------------
# Part 3 - Making the predictions and visualising the results

### Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

"""### Getting the predicted stock price of 2017"""
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

dataset_train.shape

dataset_test.shape

dataset_test['Open'].values

inputs

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

"""### Visualising the results"""

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()