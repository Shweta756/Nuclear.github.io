from sklearn.neural_network import MLPClassifier,MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense,Activation
import csv


rs=pd.read_csv("Malatya-Meterology-Data.csv")
# Read the CSV file
df = pd.read_csv('Malatya-temp1.csv')

# Select the x and y values
x_values = df['Year']
y_values = df['4']

# Create the line plot
plt.plot(x_values, y_values)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Mean temperature variation 2010-2025')

# Display the plot
plt.show()

X=rs.iloc[:,1:3]
Y=rs.iloc[:,-1:]
print(X.head)
print(Y.head)
Y=np.ravel(Y)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,random_state=1)

NN=MLPRegressor(max_iter=30,activation="relu", hidden_layer_sizes=(10,10))
MLPRegressor
NN.fit(Xtrain,Ytrain)
NN_pred=NN.predict(Xtest)
print("MSE",mean_squared_error(Ytest,NN_pred))

model=Sequential()
model.add(Dense(100, input_dim=Xtrain.shape[1], activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=["mae"])
history=model.fit(Xtrain,Ytrain,epochs=5)
model.evaluate(Xtest,Ytest)
model.save('C:/sijin/PROJECTS/OPENCV/openenv1/R-S-M/Malatya-model.keras')
print("The reduction in loss is:")
print(history.history['loss'])
plt.plot(history.history['loss'])
plt.show()