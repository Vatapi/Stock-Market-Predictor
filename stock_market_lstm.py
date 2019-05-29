import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Flatten

dataset = pd.read_csv("EOD-MSFT.csv")
dataset = dataset.reindex(index = dataset.index[::-1])
# print(dataset.head())
# print(dataset.tail())
df = dataset.iloc[ : ,4:5].values
print(df.shape)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_transformed = sc.fit_transform(df)
split = int(len(df_transformed)*0.80)
df_transformed_train = df_transformed[:split]
df_transformed_test = df_transformed[split:]

x_train = []
y_train = []
for i in range(30,len(df_transformed_train)):
    x_train.append(df_transformed_train[i-30:i , 0])
    y_train.append(df_transformed_train[i , 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
# print(x_train)
# print(y_train)
print(x_train.shape)
print(y_train.shape)

x_train = np.reshape(x_train , (x_train.shape[0] , x_train.shape[1] , 1))
# print(x_train)
print(x_train.shape)

print("Defining model....")
#Defining the Model
model = Sequential()

model.add(LSTM(units=50 , return_sequences=True , input_shape = ( x_train.shape[1] , 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60 , return_sequences=True ))
model.add(Dropout(0.2))

model.add(LSTM(units=70 , return_sequences=True ))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(units=1))

model.compile(optimizer='adam' , loss='mean_squared_error' , metrics=['accuracy'])
print(x_train.shape)
model.fit(x_train , y_train , batch_size=30 , epochs=100)

# #Predicting results
total_dataset = np.concatenate((df_transformed_train , df_transformed_test) , axis=0)
inputs = total_dataset[len(total_dataset) - len(df_transformed_test) - 30:]
inputs = inputs.reshape(-1,1)
# inputs = sc.transform(inputs) #double Transforming avoided

x_test = []
for i in range(30 , len(inputs)):
    x_test.append(inputs[i-30:i , 0])
x_test = np.array(x_test)

x_test = np.reshape(x_test , (x_test.shape[0] ,x_test.shape[1] , 1))
pred_values = model.predict(x_test)
pred_values = sc.inverse_transform(pred_values)
pred_values = np.concatenate((sc.inverse_transform(df_transformed_train) , pred_values))
print(pred_values)

#Visualizing results
plt.plot(dataset.iloc[: , 4:5].values , color='black' , label = 'Original')
plt.plot(pred_values , color='red' , label = 'Predicted')
plt.legend(loc='best')
plt.title('Predicted vs Original')
plt.show()
