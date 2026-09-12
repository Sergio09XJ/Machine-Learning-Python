import numpy as np
from sklearn.linear_model import LinearRegression

#Valores de X(entradas) para el aprendizaje; X es una matriz 2x1
horas_estudio_X = np.array([[1], [2], [3], [4], [5]])
#Valores y(objetivo) para el error; y es un vector de una dimensión
calificacion_y = np.array([50, 55, 65, 70, 80])

#Instanciamos el modelo. 
modelo = LinearRegression()

#Ponemos a entrenar al modelo, aquí el modelo itero, modificando el peso y el sesgo con calculos matemáticos, reduciendo el error. 
modelo.fit(horas_estudio_X, calificacion_y)

#Ponemos a trabajar/Predecir al modelo no es necesario y ya que es lo que buscamos encontrar, si no no tendría sentido. 
horas_estudio_nuevas_X = np.array([[6],[7]])

#Predicción, aquí el modelo la función f(X) para encontrar el y equivalente a X, aquí no se modifica el modelo, se usa. 
print(f"\n Predicciónes del Modelo: {modelo.predict(horas_estudio_nuevas_X)}")

#Los pesos de X(Como influye X en y).
print(f"\n Pesos/Pendiente que encontro en el entrenamiento: {modelo.coef_}")

#La intersección de X cuando vale 0. 
print(f"\n Sesgo/Intersección que encontro en el entrenamiento: {modelo.intercept_ }")
