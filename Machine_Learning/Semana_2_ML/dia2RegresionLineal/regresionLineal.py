from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([3, 5, 7, 9, 11, 13])

modelo = LinearRegression()

modelo.fit(X, y)

print("\n Parte 1 del ejercicio ------------------------ ")
#El intercepto es el valor inicial de y cuando X = 0 por otro la pendiente es el cambio que se da para cada X para obtener a y
print(f" Imprimimos el intercepto - Beta_0: {modelo.intercept_} \n Imprimimos la pendiente Beta_1: {modelo.coef_} \n Predcción con X = 10: {modelo.predict([[10]])}")

print("\n Parte 2 del ejercicio ------------------------ ")

Datos = pd.DataFrame()

Datos["Real"] = np.array([12, 13, 14, 15, 16])
Datos["Predicción"] = np.array([modelo.predict([[12]]), modelo.predict([[13]]), modelo.predict([[14]]), modelo.predict([[15]]), modelo.predict([[16]])])
Datos["Residuo"] = np.array([25 - modelo.predict([[12]]), 27 - modelo.predict([[13]]), 29 - modelo.predict([[14]]), 31 - modelo.predict([[15]]), 33 - modelo.predict([[16]])])

print(Datos)

print("\n Parte 3 del ejercicio ------------------------ ")

X_nuevo = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
    [5, 50],
    [6, 60]
])

y_nuevo = y

modelo.fit(X_nuevo, y_nuevo)
print(f" Imprimimos el intercepto - Beta_0: {modelo.intercept_} \n Imprimimos la pendiente Beta_1: {modelo.coef_} \n Predcción con X = [7, 70]: {modelo.predict([[7, 70]])}")

