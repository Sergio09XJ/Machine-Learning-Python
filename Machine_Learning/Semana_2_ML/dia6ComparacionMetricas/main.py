
from src.services.analizador_Modelo_Fraude import preparacionDatos, preparaciónTrainyTest, generacionModelo, entrenamiento, predicciónMuestra, generacionModeloRandomForest, comparacion
from src.services.analizador_grafico import prediccionDatos 

def main():

 print(" ------------------------------ Analizador contra Fraude ------------------------------ ")

 X, y = preparacionDatos()

 X_train, X_test, y_train, y_test = preparaciónTrainyTest(X, y)
 print(f"Datos de entrenamiento X: \n {X_train}")
 print(f"Datos de entrenamiento y: {y_train}")

 modeloClasificacion = generacionModelo()
 modeloForest  = generacionModeloRandomForest()

 modeloClasificacionEntrenado = entrenamiento(X_train, y_train, modeloClasificacion)

 modeloClasificacionEntrenadoBosque = entrenamiento(X_train, y_train, modeloForest)
 print("\n Los datos usando el primer modelo: ")
 y_test, y_sombrero = predicciónMuestra(X_test, y_test, modeloClasificacionEntrenado)

 print("\n Los datos usando el segundo modelo: ")
 y_test, y_sombrero  = predicciónMuestra(X_test, y_test, modeloClasificacionEntrenadoBosque)


 comparacion(X_test, y_test, modeloClasificacionEntrenado, modeloClasificacionEntrenadoBosque)




main()