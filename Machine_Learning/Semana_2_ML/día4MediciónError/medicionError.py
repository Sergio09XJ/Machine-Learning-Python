from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt

import numpy as np

y_real = np.array([50, 60, 70, 80, 90, 100])

y_test = np.array([52, 58, 71, 77, 93, 96])

mae = mean_absolute_error(y_real, y_test)
mse = mean_squared_error(y_real, y_test)
r2score = r2_score(y_real, y_test)

#Son las unidades de error que tiene el modelo respecto a los valores reales. 
print(f"Imprimimos el error promedio en las unidades del problema: {mae} unidades ")

#Son las unidades de error que tiene el modelo respecto a los v. reales al cuadrado, aquí es muy sensible a los outlines(elementos de y que estan disparados respecto a los demas).
print(f"Imprimimos el error promedio al cuadrado del problema: {mse} unidades^2 ")

#Es la precisión que tiene el modelo para su capacidad de predecir. 
print(f"Imprimimos la capacidad de caos(varianza) eliminado del modelo para dar resultados precisos: {r2score} ")

#Ejercicio 2 -----------------------------------

y_real_2 = np.array([50, 60, 70, 80, 90])

y_test_2 = np.array([52, 58, 71, 77, 93])

maeDos = mean_absolute_error(y_real_2, y_test_2)
mseDos = mean_squared_error(y_real_2, y_test_2)
r2scoreDos = r2_score(y_real_2, y_test_2)

print("\n Ejercicio 2 ----------------------------- \n")
#Son las unidades de error que tiene el modelo respecto a los valores reales. 
print(f"Imprimimos el error promedio en las unidades del problema: {maeDos} unidades ")

#Son las unidades de error que tiene el modelo respecto a los v. reales al cuadrado, aquí es muy sensible a los outlines(elementos de y que estan disparados respecto a los demas).
print(f"Imprimimos el error promedio al cuadrado del problema: {mseDos} unidades^2 ")

#Es la medida que permite entender cuanto caos logro reducir(entender) 
print(f"Imprimimos la capacidad de caos(varianza) eliminado del modelo para dar resultados precisos: {r2scoreDos} ")


y_test_2punto2 = [52, 58, 71, 77, 120]

maeDosPuntoDos = mean_absolute_error(y_real_2, y_test_2punto2)
mseDosPuntoDos = mean_squared_error(y_real_2, y_test_2punto2)
r2scoreDosPuntoDos = r2_score(y_real_2, y_test_2punto2)

print("\n Parte 2 ejercicio 2 ------------------------------ ")

#Son las unidades de error que tiene el modelo respecto a los valores reales. 
print(f"Imprimimos el error promedio en las unidades del problema: {maeDosPuntoDos} unidades ")

#Son las unidades de error que tiene el modelo respecto a los v. reales al cuadrado, aquí es muy sensible a los outlines(elementos de y que estan disparados respecto a los demas).
print(f"Imprimimos el error promedio al cuadrado del problema: {mseDosPuntoDos} unidades^2 ")

#Es la medida que permite entender cuanto caos logro reducir(entender) 
print(f"Imprimimos la capacidad de caos(varianza) eliminado del modelo para dar resultados precisos: {r2scoreDosPuntoDos} ")

#La razón por la que el segundo tiene menor precisión al predecir es debido a l 120, esto aumenta la varianza respecto a los demas. 
#Haciendo que se jale la recta hacía ese punto, aquí se debe de trabajar con ese dato para lograr el objetivo. 


#Gráfica de y real vs y predicta. 

plt.scatter(y_real_2,y_test_2punto2)

plt.xlabel("Los datos reales.")
plt.ylabel("Los datos predichos.")

plt.title("Datos reales vs Predichos.")
plt.show()