from sklearn.model_selection import train_test_split #Para dividir en entrenamiento y test
from sklearn.linear_model import LogisticRegression #Para el modelo de clasificación
from sklearn.metrics import accuracy_score # Para medir qué tan precisa es la clasificación del modelo
import  numpy as np

#Hora de registro de inventario con cantidad de productos. 
X = np.array([[1,10], [1,15], [3,20], [5,40], [6,50], [8,70]]) #Features 

#Necesita automatización o no. 
y = np.array([0,0,0,1,1,1]) #Targets

#Dividimos y guardamos los train y los tests
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

print(f"\nFeatures de entrenamiento: {X_train}")
print(f"\n Features para trabajo del modelo: {X_test}")
print(f"\n Target de entrenamiento: {y_train}")
print(f"\n Target objetivo: {y_test}")

modeloClasificacion = LogisticRegression() #Instancia del modelo

modeloClasificacion.fit(X_train, y_train) #Entrenamiento

prediccion = modeloClasificacion.predict(X_test) #Trabajamos el modelo con Xtest

print(f"\nImprimimos la predicción: {prediccion}") #Imprimimos la predicción
print(f"\nImprimimos los valores y test: {y_test}") #Imprimimos los valores reales. 

print(f"\nMedimos con accuracy: {accuracy_score(y_test, prediccion)}") #Imprimimos la presición del modelo. 