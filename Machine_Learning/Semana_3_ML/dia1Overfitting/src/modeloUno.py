from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error)

import pandas as pd
import numpy as np

from pathlib import Path 
raiz_proyecto = Path(__file__).resolve().parent.parent

tablaNolineal = pd.read_csv(raiz_proyecto / "data" / "tablavalores.csv")


print(tablaNolineal)
print(f"\nEl promedio es de : {tablaNolineal.to_numpy().mean()}")

X = tablaNolineal.drop(columns=["feature_8"]).to_numpy()
y = tablaNolineal[["feature_8"]].to_numpy().reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


poly = PolynomialFeatures(degree= 1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

modelo = LinearRegression()
modelo.fit(X_train_poly, y_train)

predicción_test = modelo.predict(X_test_poly)
predicción_train = modelo.predict(X_train_poly)

mse_test = mean_squared_error(y_test, predicción_test)
mse_train = mean_squared_error(y_train, predicción_train)

print(f"La predicción es: \n{predicción_test}")
print(f"Los datos reales son es: \n{y_test}")

print(f"\nEl MSE con los datos de entrenamiento es: {mse_train}")
print(f"\nEl MSE con los datos de prueba es: \n{mse_test}")

