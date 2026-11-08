
from src.services.analizador_Modelo_Fraude import preparacionDatos, preparaciónTrainyTest, generacionModeloRegression, generacionModeloArbolDesicion, entrenamiento, predicción, evaluacion
from src.services.analizador_grafico import prediccionDatosGrafico 

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
 modeloRandomForest = generacionModeloArbolDesicion(200, 10, 5, 2,'sqrt', True, 42, -1)

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






main()