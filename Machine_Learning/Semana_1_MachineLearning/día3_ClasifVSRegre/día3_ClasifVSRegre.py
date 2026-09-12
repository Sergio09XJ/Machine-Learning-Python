import numpy as np

#Para clasificación
from sklearn.linear_model import LogisticRegression

#Para predicción númerica 
from sklearn.linear_model import LinearRegression


# Clasificación 

# Target de visitas de app y compras previas. 
Features = np.array([[1, 0], [2,0], [2,1], [4,1], [5,2], [6,4]])

Target = np.array([0, 0, 0, 1, 1, 1])

modelo = LogisticRegression()

modelo.fit(Features, Target)

nuevoCliente_Target = np.array([[5, 2]])
print(f"\nEl modelo nos die que el cliente: {"compro" if modelo.predict(nuevoCliente_Target) == 1  else "no compro"} \n El peso quedo como: {modelo.coef_} \n y el sesgo como: {modelo.intercept_}")

#Predicción

#Target de clientes y  publicidad
X = np.array([[50, 1000], [60, 1200], [70, 1500], [80, 1800], [90, 2000], [100, 2300]])

#Ventas que se tuvieron relacionado al target
y = np.array([10000, 11500, 13000, 14500, 16000, 18000])

modeloReg = LinearRegression()

modeloReg.fit(X, y)

nuevoDato = np.array([[85, 1900]])
print(f"\nEl modelo nos die que el cliente tuvo ventas de: {modeloReg.predict(nuevoDato)} \n El peso quedo como: {modeloReg.coef_} \n y el sesgo como:  {modeloReg.intercept_}")
