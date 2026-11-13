
from src.services.analizador_Modelo_Fraude import preparacionDatos, preparaciónTrainyTest, generacionModeloBosqueRandom, entrenamiento, predicción, MinMaxEscalado, escaladoEstadar, generacionModeloVecinos, accuracy_scoreFraude
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler #Métodos para escalar valores. 
import numpy as np
import pandas as pd
def main():

 print("------------------------------ Analizador contra Fraude ------------------------------ ")

 #Se paramos los datos para el entrenamiento, validacion y pruebas: 

 X, y = preparacionDatos()

 #Separamos los datos para el entrenamiento y la validación
 X_aux, X_test, y_aux, y_test = preparaciónTrainyTest(X, y, 0.15)
 X_train, X_validacion, y_train, y_validacion = preparaciónTrainyTest(X_aux, y_aux, 0.15)

 #instanciamos el objeto Pipeline
 pipe = Pipeline([
       ("scaler", StandardScaler()),
       ("modeloKNN",generacionModeloVecinos())
   ]
  )

 #Entrenamos usando el modelo con pipe
 pipe.fit(X_train, y_train)

 #predecimos la validación y el test con pipe
 valoresPredichosVal = pipe.predict(X_validacion)
 valoresPredichosTest = pipe.predict(X_test)

 #Imprimimos el rendimiento de los modelos. 
 print(f"\n -----------------  Comparación del Accuracy  ----------------- ")
 print(f"El Accuracy para la validación es: {accuracy_scoreFraude(y_validacion, valoresPredichosVal)}")
 print(f"El Accuracy para el test es: {accuracy_scoreFraude(y_test, valoresPredichosTest)}")

 #Entreamiento del modelo de forma manual: 

 #Instanciamos y entrenamos el modelo: 
 modelo = generacionModeloVecinos()

 #instanciamos el escalador de datos: 
 scaler = StandardScaler()

 #Entrenamos el modelo 
 modelo.fit(X_train, y_train)

 #Validación del modelo 
 y_sombreroValidacion = modelo.predict(X_validacion)

 #Testeamos el modelo 
 y_sombreroTest = modelo.predict(X_test)

 #Medimos el accuracy 

 acurracyValidacion = accuracy_scoreFraude(y_validacion, y_sombreroValidacion)
 acurracyTest= accuracy_scoreFraude(y_test, y_sombreroTest)

 print(f"\n -----------------  Pipeline / Manual  ----------------- ")
 print(f"El Accuracy para la validación en PipeLine es: {accuracy_scoreFraude(y_validacion, valoresPredichosVal)}")
 print(f"El Accuracy para el test en PipeLine es: {accuracy_scoreFraude(y_test, valoresPredichosTest)}")
 print(f"El Accuracy para la validación en Manual es: {acurracyValidacion}")
 print(f"El Accuracy para el test en Manual es: {acurracyTest}")














main()