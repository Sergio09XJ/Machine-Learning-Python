
#Modelo de IA
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score #Para test 
from sklearn.preprocessing import MinMaxScaler, StandardScaler #Métodos para escalar valores. 
from sklearn.neighbors import KNeighborsClassifier #Modelo basado en vecinos. 
from sklearn.linear_model import LogisticRegression  #Modelo de regression
from sklearn.ensemble import RandomForestClassifier #Modelo de arboles de decisión
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score,recall_score, accuracy_score #Conocer la exactutud del modelo
from src.services.analizador_Datos import toDataFrame, limpieza    # para el DataFrame de datos. 
from sklearn.ensemble import RandomForestClassifier #Modelo de arboles de decisión

import numpy as np # para de forma optimizada arrays

def preparacionDatos(): 

  DatosDataFrame = toDataFrame(0)
  DatosDataFrame = limpieza(DatosDataFrame, 0)
  #Datos De entrenamiento 
  X_Features = np.array(DatosDataFrame.iloc[:, 2:8])  
  y_target = np.array(DatosDataFrame["is_fraud"])
  return X_Features, y_target

def preparaciónTrainyTest(X, y, porcentaje):
   #Separación para train y test
   return train_test_split(X, y, test_size=porcentaje, random_state=42, stratify=y)

def generacionModeloRegression(): 
   return LogisticRegression()

def generacionModeloArbolDesicion(maxdepth):
   return DecisionTreeClassifier(max_depth=maxdepth)

def generacionModeloBosqueRandom(cantArbol, maxProf, cantDiv, cantHoja, features, remplazo, randomState, nucleos):
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

def generacionModeloVecinos():
   return KNeighborsClassifier()

def escaladoEstadar(escalaEstandar, fit, datos): 
   if fit == 1:
      datosEscalados =  escalaEstandar.fit_transform(datos)
      return datosEscalados
   else: 
      return escalaEstandar.transform(datos)

def MinMaxEscalado(MinMaxEscalado, fit, datos): 
   if fit == 1:
      return MinMaxEscalado.fit_transform(datos)
   else: 
      return MinMaxEscalado.transform(datos)
   

def entrenamiento(X_train, y_train, modelo): 
  #Entrenamiento
  modelo.fit(X_train, y_train)
  return modelo

def predicción(modelo, X_test):
  return modelo.predict(X_test)

def accuracy_scoreFraude(y_test, y_sombrero): 
   return accuracy_score(y_test, y_sombrero)

     

def evaluacion(y_sombrero, y_test): 
    print(f"\nImprimimos y_test: {y_test} \nImprimimos y_predicción: {y_sombrero}")
    print(f"\nImprimimos la exactitud del modelo \n {accuracy_score(y_test, y_sombrero)}")
    print(f"Si es fraude es verdad: {precision_score(y_test, y_sombrero)}")
    print(f"Cantidad de fraudes reales es: { recall_score(y_test, y_sombrero)}")
    print(f"El f1 score del modelo es: {f1_score(y_test, y_sombrero)}")

def validacion_cruzada(modelo, X_train, y_train, folds, scoring, tipoorg=None): 
   
   if tipoorg == 1: 
      cv_estratificado = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
      return cross_val_score(modelo,X_train, y_train, cv=cv_estratificado, scoring=scoring)
   else: 
      return cross_val_score(modelo,X_train, y_train, cv=folds, scoring=scoring)














