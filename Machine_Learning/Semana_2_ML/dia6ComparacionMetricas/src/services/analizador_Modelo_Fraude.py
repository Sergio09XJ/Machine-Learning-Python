
#Modelo de IA
from sklearn.model_selection import train_test_split #Para test 
from sklearn.linear_model import LogisticRegression  #Modelo de regression
from sklearn.ensemble import RandomForestClassifier #modelo de clasificación 
 #conocer la exactitud del modelo 
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix)   
      
from src.services.analizador_Datos import toDataFrame, limpieza    # para el DataFrame de datos. 

import numpy as np # para de forma optimizada arrays

def preparacionDatos(): 

  DatosDataFrame = toDataFrame()
  DatosDataFrame = limpieza(DatosDataFrame)
  #Datos De entrenamiento 
  X_Features = DatosDataFrame[["amount","transactions_last_24h","distance_from_home_km","is_foreign_country","is_high_risk_category","device_score"]].to_numpy()
  y_target = np.array(DatosDataFrame["is_fraud"])
  return X_Features, y_target

def preparaciónTrainyTest(X, y):
   #Separación para train y test
   return train_test_split(X, y, test_size=0.4, random_state=42)
   #X_train, X_test, y_train, y_test 

def generacionModelo(): 
   return LogisticRegression()

def generacionModeloRandomForest():
   return RandomForestClassifier()
   

def entrenamiento(X_train, y_train, modelo): 
  #Entrenamiento
  modelo.fit(X_train, y_train)
  return modelo


def predicciónMuestra(X_test, y_test, modelo): 
  y_sombrero = modelo.predict(X_test)
  print(f"\nImprimimos y_test: {y_test} \nImprimimos y_predicción: {y_sombrero}")
  print(f"\nImprimimos la exactitud del modelo \n {accuracy_score(y_test, y_sombrero)}")
  print(f"\nImprimimos cuales TP son realmente TP (precision): {precision_score(y_test, y_sombrero)}")
  print(f"\nImprimimos Cuantos TP logro encontrar el modelo (recall): {recall_score(y_test, y_sombrero)}")
  print(f"\nImprimimos la relación entre precision y recall(f1_score): {f1_score(y_test, y_sombrero)}")
  print(f"\nImprimimos el tipo de aciertos y errores (conf_matrix): {confusion_matrix(y_test, y_sombrero)}")
  return y_test, y_sombrero







def comparacion(X_test, y_test, modelo, modeloDos): 
  y_sombrero    = modelo.predict(X_test)
  y_sombreroUno = modeloDos.predict(X_test)

  accUno = accuracy_score(y_test, y_sombrero)
  accDos = accuracy_score(y_test, y_sombreroUno)

  recallUno = recall_score(y_test, y_sombrero)
  recallDos = recall_score(y_test, y_sombreroUno)

  precisionUno = precision_score(y_test, y_sombrero)
  precisionDos = precision_score(y_test, y_sombreroUno)

  f1_scoreUno = f1_score(y_test, y_sombrero)
  f1_scoreDos = f1_score(y_test, y_sombreroUno)
  print("\n ========================================================================= ")
  print("                           COMPARACIÓN DE MODELOS                          ")
  print(" ========================================================================= ")
  print("                         Accuracy | Precision  | Recall |  F1   ")  
  print(f" LogisticRegression    {round(accUno, 3) * 100}%  | {round(recallUno, 3) * 100}% | {round(precisionUno, 3) * 100}%  | {round(f1_scoreUno, 3) * 100}% ") 
  print(f" RandomForest          {round(accDos,3) * 100}%   | {round(recallDos, 3) * 100}% | {round(precisionDos, 3) * 100}%  | {round(f1_scoreDos, 3) * 100}% ")   
  print(" ========================================================================= ") 
  print("\n")                          
                      
        
