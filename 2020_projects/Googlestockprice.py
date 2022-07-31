import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.read_csv("Google.csv", date_parser= True)
data=data.dropna()
data.isnull()
data1 = pd.read_csv("Google.csv")
data1.tail()


training_data = data[data["Date"]<"2019-01-01"].copy()
training_set = data[data["Date"]<"2019-01-01"].copy()

test_data = data[data["Date"]>="2019-01-01"].copy()
test_set = data[data["Date"]>="2019-01-01"].copy()

training_set = training_set.drop(["Date", "Adj Close"], axis=1)

scaler = MinMaxScaler()


training_set = scaler.fit_transform(training_set)


X_train = []
y_train=[]


training_set.shape[0]

for i in range(60, training_set.shape[0]):
    X_train.append(training_set[i-60:i])
    y_train.append(training_set[i, 0])


X_train, y_train = np.array(X_train), np.array(y_train)

X_train.shape, y_train.shape


##BUILDING LSTM

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


regressior = Sequential()

regressior.add(LSTM(units =60,activation = "relu", return_sequences = True,
                    input_shape =(X_train.shape[1],X_train.shape[2])))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units =60,activation = "relu", return_sequences = True))
regressior.add(Dropout(0.3))

regressior.add(LSTM(units =80,activation = "relu", return_sequences = True))
regressior.add(Dropout(0.4))

regressior.add(LSTM(units =120,activation = "relu"))
regressior.add(Dropout(0.5))


regressior.add(Dense(units =1))

regressior.summary()


regressior.compile(optimizer = "adam", loss= "mean_squared_error")

history = regressior.fit(X_train, y_train, batch_size = 32, epochs = 30) 
                    

###PREPARE TEST DATASET

past_60_days = training_data.tail(60)

df = past_60_days.append(test_data, ignore_index=True)

df= df.drop(["Date", "Adj Close"], axis=1)


inputs = scaler.fit_transform(df)

X_test=[]
y_test=[]

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])



X_test, y_test = np.array(X_test), np.array(y_test)

X_test.shape, y_test.shape


y_pred = regressior.predict(X_test)

scaler.scale_

scale = 1/8.85034074e-05

y_pred = y_pred*scale

y_test = y_test*scale

##VISUALIZATION

%matplotlib
#Plot training and validation accuracy values
plt.figure(figsize=(14, 5))
plt.plot(y_test, color="red", label= "Real Google Stock Price")
plt.plot(y_pred, color = "Blue", label = "Predicted Google Stock Price")
plt.title("Google stock price prediction")
plt.ylabel("Google stock price")
plt.grid()
plt.xlabel("Time")
plt.legend(loc= "upper left")
plt.show()

















