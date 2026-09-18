from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np

horasEstudio = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
calificaciones = np.array([ 50,

    55,

    60,

    65,

    70,

    75,

    80,

    85,

    90,

    95])

modelo = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(horasEstudio, calificaciones, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)

print(f"Resultados reales: {y_test}")
print(f"Resultados predichos: {modelo.predict(X_test)}")

residuos = np.array([y_test - modelo.predict(X_test)])

print(f"Imprimimos los residuos: {residuos}")

plt.scatter(y_test, modelo.predict(X_test))

plt.xlabel("Valor Real")
plt.ylabel("Valor predicho")
plt.title("Valor Real vs Valor Predicho")

plt.show()

plt.scatter(modelo.predict(X_test), residuos)

plt.axhline(0)

plt.xlabel("Valor predicho")
plt.ylabel("Residuos")

plt.title("Valor Predicho vs Residuos ")

lt.show()
