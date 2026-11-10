
from src.services.analizador_Modelo_Fraude import preparacionDatos, preparaciónTrainyTest, generacionModeloRegression, generacionModeloBosqueRandom ,generacionModeloArbolDesicion, entrenamiento, predicción, accuracy_scoreFraude
from src.services.analizador_grafico import prediccionDatosGrafico 
import numpy as np
import pandas as pd
def main():

 print("------------------------------ Analizador contra Fraude ------------------------------ ")

 #Se paramos los datos para el entrenamiento, validacion y pruebas: 
 X, y = preparacionDatos()

 X_aux, X_test, y_aux, y_test = preparaciónTrainyTest(X, y, 0.15)
 X_train, X_validación, y_train, y_validación = preparaciónTrainyTest(X_aux, y_aux, 0.15)
 print(f"Datos de entrenamiento X: \n {X_train}")
 print(f"\nDatos de entrenamiento y: \n {y_train}")

 #Configuramos los modelos así como los hiperparametros
 modeloClasificacion = generacionModeloRegression()
 modeloRandomForest = generacionModeloBosqueRandom(200, 10, 5, 2,'sqrt', True, 42, -1)
 modeloSimple = generacionModeloArbolDesicion(1)
 modeloMedio = generacionModeloArbolDesicion(2)
 modeloComplejo = generacionModeloArbolDesicion(None)

 arrayScoreEntrenado = []
 arrayScorePruebas = []
 arrayScoreValidacion = []

 # ---------------------------------------------- Modelo Simple ---------------- 
 #Entrenamiento: 

 modeloSimpleEntrenado = entrenamiento(X_train, y_train, modeloSimple)
 print(f"\nProfundidad del modelo Simple : {modeloSimpleEntrenado.get_depth()}")
 y_sombreroEntrenamientoSimple = predicción(modeloSimpleEntrenado, X_train) 
 arrayScoreEntrenado.append(accuracy_scoreFraude(y_sombreroEntrenamientoSimple, y_train))

 # Pruebas: 

 y_sombreroPruebasSimple = predicción(modeloSimpleEntrenado, X_test) 
 arrayScorePruebas.append(accuracy_scoreFraude(y_sombreroPruebasSimple, y_test))

 # Validación: 

 y_sombreroValidacionSimple = predicción(modeloSimpleEntrenado, X_validación) 
 arrayScoreValidacion.append(accuracy_scoreFraude(y_sombreroValidacionSimple, y_validación))

  # ---------------------------------------------- Modelo Medio ---------------- 
 #Entrenamiento: 

 modeloMedioEntrenado = entrenamiento(X_train, y_train, modeloMedio)
 print(f"\nProfundidad del modelo Medio : {modeloMedioEntrenado.get_depth()}")
 y_sombreroEntrenamientoMedio = predicción(modeloMedioEntrenado, X_train) 
 arrayScoreEntrenado.append(accuracy_scoreFraude(y_sombreroEntrenamientoMedio, y_train))

 # Pruebas: 

 y_sombreroPruebasMedio = predicción(modeloMedioEntrenado, X_test) 
 arrayScorePruebas.append(accuracy_scoreFraude(y_sombreroPruebasMedio, y_test))

 # Validación: 

 y_sombreroValidacion = predicción(modeloMedioEntrenado, X_validación) 
 arrayScoreValidacion.append(accuracy_scoreFraude(y_sombreroValidacion, y_validación))

  # ---------------------------------------------- Modelo Complejo ---------------- 
 #Entrenamiento: 

 modeloComplejoEntrenado = entrenamiento(X_train, y_train, modeloComplejo)
 print(f"\nProfundidad del modelo Complejo : {modeloComplejoEntrenado.get_depth()}")
 y_sombreroEntrenamientoComplejo = predicción(modeloComplejoEntrenado, X_train) 
 arrayScoreEntrenado.append(accuracy_scoreFraude(y_sombreroEntrenamientoComplejo, y_train))

 # Pruebas: 

 y_sombreroPruebasComplejo = predicción(modeloComplejoEntrenado, X_test) 
 arrayScorePruebas.append(accuracy_scoreFraude(y_sombreroPruebasComplejo, y_test))

 # Validación: 

 y_sombreroValidacion = predicción(modeloComplejoEntrenado, X_validación) 
 arrayScoreValidacion.append(accuracy_scoreFraude(y_sombreroValidacion, y_validación))

 nombre = ["Modelo Simple", "Modelo Normal", "Modelo Complejo"]
 dataframeExactud = pd.DataFrame({
   "Modelo" : nombre, 
   "Exactitud_Entrenamiento" : arrayScoreEntrenado ,
   "Exactitud_Pruebas" : arrayScorePruebas,
   "Exactitud_Validación" :  arrayScoreValidacion 
 })
 
 pd.set_option('display.max_columns', None)
 pd.set_option('display.width', 1000)
 print(dataframeExactud.to_string())



"""
 #Entrenamiento: 
 print("\n -------------------- Entrenamiento -------------------- ")
 modeloClasificacionEntrenado = entrenamiento(X_train, y_train, modeloClasificacion)
 y_sombreroEntrenamiento = predicción(modeloClasificacionEntrenado, X_train) 
 evaluacion(y_sombreroEntrenamiento, y_train)

 # Pruebas: 
 print("\n -------------------- Pruebas -------------------- ")
 y_sombreroPruebas = predicción(modeloClasificacionEntrenado, X_test) 
 evaluacion(y_sombreroPruebas, y_test)

 # Validación: 
 print("\n -------------------- Validación -------------------- ")
 y_sombreroValidacion = predicción(modeloClasificacionEntrenado, X_validación) 
 evaluacion(y_sombreroValidacion, y_validación)

  #Entrenamiento: 
 print("\n -------------------- Entrenamiento Random Forest -------------------- ")
 modeloRandomForestEntrenado  = entrenamiento(X_train, y_train, modeloRandomForest)
 y_sombreroEntrenamiento = predicción(modeloRandomForestEntrenado, X_train) 
 evaluacion(y_sombreroEntrenamiento, y_train)

 # Pruebas: 
 print("\n -------------------- Pruebas Random Forest -------------------- ")
 y_sombreroPruebas = predicción(modeloRandomForestEntrenado, X_test) 
 evaluacion(y_sombreroPruebas, y_test)

 # Validación: 
 print("\n -------------------- Validación Random Forest -------------------- ")
 y_sombreroValidacion = predicción(modeloRandomForestEntrenado, X_validación) 
 evaluacion(y_sombreroValidacion, y_validación)
"""





main()