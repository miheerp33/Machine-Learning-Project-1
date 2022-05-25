
import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
url = "winequality-red.csv"
names = ['total sulfur dioxide','fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'residual sugar',
         'pH',  'density', 'free sulfur dioxide', 'alcohol', 'sulphates', 'quality']
dataset = read_csv(url, usecols=names)

array = dataset.values

X = array[:,0:10]
y = array[:,10]


X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


for i in range(len(predictions)):
    print(predictions[i], '-||-', Y_validation[i])

score = model.score(X_validation, Y_validation)
print(score*100, '%')
plotter = X_validation[:,1]
plt.scatter(plotter, Y_validation, color='black')
#plt.plot(plotter, predictions, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

