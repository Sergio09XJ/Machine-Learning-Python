from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path 
raiz_proyecto = Path(__file__).resolve().parent.parent

tablaNolineal = pd.read_csv(raiz_proyecto / "data" / "tablavalores.csv")


print(tablaNolineal)
print(f"\nEl promedio es de : {tablaNolineal.to_numpy().mean()}")

X = tablaNolineal.drop(columns=["feature_3"]).to_numpy()
y = tablaNolineal[["feature_3"]].to_numpy().reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


poly = PolynomialFeatures(degree= 4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

modelo = LinearRegression()
#modelo.fit(X_train, y_train)
modelo.fit(X_train_scaled, y_train)



predicción_test = modelo.predict(X_test_scaled)
predicción_train = modelo.predict(X_train_scaled)


mse_test = mean_squared_error(y_test, predicción_test)
mae_test = mean_absolute_error(y_test, predicción_test)

mse_train = mean_squared_error(y_train, predicción_train)
mae_train = mean_absolute_error(y_train, predicción_train)

r2_train = r2_score(y_train, predicción_train)
r2_test  = r2_score(y_test, predicción_test)

print(f"La predicción es: \n{predicción_test}")
print(f"Los datos reales son es: \n{y_test}")

print(f"\nEl MSE con los datos de entrenamiento es: {mse_train}")
print(f"\nEl MSE con los datos de prueba es: \n{mse_test}")

print(f"\nEl R^2 con los datos de entrenamiento es: {r2_train}")
print(f"\nEl R^2 con los datos de prueba es: \n{r2_test}")

print(f"\nEl MAE con los datos de entrenamiento es: {mae_train}")
print(f"\nEl MAE con los datos de prueba es: \n{mae_test}")

# Línea diagonal ideal (donde Y_real == Y_predicho)
plt.scatter(X_train[:, 0], y_train, label="Datos reales", alpha=0.7)
plt.scatter(X_train[:, 0], predicción_train, label="Predicciones", color="red", alpha=0.7)

plt.xlabel("Característica 0 (feature_0)")
plt.ylabel("Y")
plt.title("Entrenamiento: valores reales vs predichos (feature_0)")
plt.legend()
plt.grid(True)


plt.savefig('/app/Semana_3_ML/dia1Overfitting/graficas/graficaDos.pdf', bbox_inches='tight')
plt.close()