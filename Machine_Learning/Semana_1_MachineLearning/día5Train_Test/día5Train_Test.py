import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_Horas = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_calificacion = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

X_train, X_test, y_train, y_test = train_test_split(X_Horas, y_calificacion, test_size=0.2, random_state=42)

print(f"\nLos features que se usan son: {X_train}")
print(f"\nLos features test que se usan son: {X_test}")
print(f"\nEl target que se usa es: {y_train}")
print(f"\nEl target test que se usa es: {y_test}")

#Creamos la instancia
modelo = LinearRegression()

#Entrenamos el modelo | Usamos X_train para que el modelo aprenda
modelo.fit(X_train, y_train)
#Es importante separar train y test ya que si el funcionamiento del modelo usamos los mismos que en train, el modelo solo repetiria lo que ya "aprendio - vio"  esto causaria un sesgo. 

#Predecimos 
prediciones = modelo.predict(X_test) # | Usamos X_test com prueba en el funcionamiento del modelo. -> Inferencía

#Comparamos | Usamos y_test solo para compara con los valores predichos, y_train es el unico que se usa para el entrenamiento. 
print(f"\nLas predicciones son {prediciones} \nMientras que los valores reales son: {y_test}")
#Las predicciones cumplen y_test por lo que el entrenamiento fue efectivo. 

#Si el modelo tuviera malos resultados en test y no en train significa que hubo un sesgo al entrenar(Se usaron los mismo valores de train al checar test y al correr con valores diferentes no aprendio bien) o  se pusieron pocos valores(features y target) de referencia en el entrenamiento. 
# O el modelo aprendio bien los datos de entrenamiento pero no aprendio a generalizar con otros datos,  o el modelo no es el indicado. 
