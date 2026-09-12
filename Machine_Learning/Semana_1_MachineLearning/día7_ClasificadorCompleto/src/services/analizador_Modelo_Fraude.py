
#Modelo de IA
from sklearn.model_selection import train_test_split #Para test 
from sklearn.linear_model import LogisticRegression  #Modelo de regression
from sklearn.metrics import accuracy_score           #conocer la exactitud del modelo 
from src.services.analizador_Datos import toDataFrame, limpieza    # para el DataFrame de datos. 

import numpy as np # para de forma optimizada arrays

def preparacionDatos(): 

  DatosDataFrame = toDataFrame()
  DatosDataFrame = limpieza(DatosDataFrame)
  #Datos De entrenamiento 
  X_Features = np.array(DatosDataFrame.iloc[0:21,2:13])
  y_target = np.array(DatosDataFrame["is_fraud"])
  return X_Features, y_target

def preparaciónTrainyTest(X, y):
   #Separación para train y test
   return train_test_split(X, y, test_size=0.4, random_state=42)

def generacionModelo(): 
   return LogisticRegression()
   

def entrenamiento(X_train, y_train, modelo): 
  #Entrenamiento
  modelo.fit(X_train, y_train)
  return modelo


def predicciónMuestra(X_test, y_test, modelo): 
  y_sombrero = modelo.predict(X_test)
  print(f"\nImprimimos y_test: {y_test} \nImprimimos y_predicción: {y_sombrero}")
  print(f"\nImprimimos la exactitud del modelo \n {accuracy_score(y_test, y_sombrero)}")
  return y_test, y_sombrero










