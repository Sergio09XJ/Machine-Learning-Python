import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.model_selection import train_test_split

ViviendaDT = pd.read_csv("datosVivienda.csv")

X = np.array(ViviendaDT.iloc[: , 0:3])
y = np.array(ViviendaDT.iloc[: , 3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

modelo = LinearRegression()

modelo.fit(X_train, y_train)

y_predicha = modelo.predict(X_test)

print(f"\n Imprimimos los valores de y test: {y_test} \n Comparamos con los valores de y_sombrero {y_predicha}")
print(f"\nImprimimos la diferencia de precio del modelo \n {mean_squared_error(y_test, y_predicha)}")
print(f"\nImprimimos la exactitud del modelo \n {r2_score(y_test, y_predicha)}")


plt.scatter(ViviendaDT.iloc[: , 0], ViviendaDT.iloc[: , 3], color="blue")
plt.title(" ---------- Grafica de metros cuadrados vs precio ---------- ")
plt.xlabel("Metros cuadrados")
plt.ylabel("Precio")
plt.grid(True)
plt.show()
