
#Modelo de IA
from sklearn.model_selection import train_test_split #Para test 
from sklearn.linear_model import LogisticRegression  #Modelo de regression
from sklearn.ensemble import RandomForestClassifier #Modelo de arboles de decisión
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score    #conocer la exactitud del modelo 
from src.services.analizador_Datos import toDataFrame, limpieza    # para el DataFrame de datos. 
from sklearn.ensemble import RandomForestClassifier #Modelo de arboles de decisión

import numpy as np # para de forma optimizada arrays

def preparacionDatos(): 

  DatosDataFrame = toDataFrame()
  DatosDataFrame = limpieza(DatosDataFrame)
  #Datos De entrenamiento 
  X_Features = np.array(DatosDataFrame.iloc[:, 2:13])  
  y_target = np.array(DatosDataFrame["is_fraud"])
  return X_Features, y_target

def preparaciónTrainyTest(X, y, porcentaje):
   #Separación para train y test
   return train_test_split(X, y, test_size=porcentaje, random_state=42, stratify=y)

def generacionModeloRegression(): 
   return LogisticRegression()

def generacionModeloArbolDesicion(cantArbol, maxProf, cantDiv, cantHoja, features, remplazo, randomState, nucleos):
   return RandomForestClassifier(
                n_estimators=cantArbol,    # Construye 200 árboles (en lugar del valor por defecto de 100)
                max_depth=maxProf,         # Limita la profundidad máxima de cada árbol a 10 niveles
                min_samples_split=cantDiv, # Pide al menos 5 muestras para dividir un nodo interno
                min_samples_leaf=cantHoja, # Pide al menos 2 muestras en cada hoja final
                max_features=features,     # Evalúa sqrt(n_features) en cada división
                bootstrap=remplazo,        # Usa muestreo con reemplazo
                random_state=randomState,  # Garantiza la reproducibilidad de los resultados 42
                n_jobs=nucleos             # Usa todos los núcleos del procesador para entrenar más rápido -1
)
   

def entrenamiento(X_train, y_train, modelo): 
  #Entrenamiento
  modelo.fit(X_train, y_train)
  return modelo

def predicción(modelo, X_test):
  return modelo.predict(X_test)



def predicciónMuestra(y_test, prediccion): 
  y_sombrero = prediccion
  print(f"\nImprimimos y_test: {y_test} \nImprimimos y_predicción: {y_sombrero}")
  print(f"\nImprimimos la exactitud del modelo \n {accuracy_score(y_test, y_sombrero)}")

def MAE(y_sombrero, y_test): 
  return mean_absolute_error(y_sombrero, y_test)

def MSE(y_sombrero, y_test): 
  return mean_squared_error(y_sombrero, y_test)

def R_2(y_sombrero, y_test): 
  return r2_score(y_sombrero, y_test)

def evaluacion(y_sombrero, y_test): 
  print(f"El promedio del error absuluto es: {MAE(y_sombrero, y_test)}")
  print(f"El promedio del error absuluto al cuadrado: { MSE(y_sombrero, y_test)}")
  print(f"El score del modelo es: {R_2(y_sombrero, y_test)}")












