
from src.services.analizador_Modelo_Fraude import preparacionDatos, preparaciónTrainyTest, generacionModelo, entrenamiento, predicciónMuestra
from src.services.analizador_grafico import prediccionDatos 

def main():

 print(" ------------------------------ Analizador contra Fraude ------------------------------ ")

 X, y = preparacionDatos()

 X_train, X_test, y_train, y_test = preparaciónTrainyTest(X, y)
 print(f"Datos de entrenamiento X: \n {X_train}")
 print(f"Datos de entrenamiento y: {y_train}")

 modeloClasificacion = generacionModelo()

 modeloClasificacionEntrenado = entrenamiento(X_train, y_train, modeloClasificacion)


 y_test, y_sombrero = predicciónMuestra(X_test, y_test, modeloClasificacionEntrenado)


 prediccionDatos(y_test, y_sombrero)




main()