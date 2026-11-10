
from src.services.analizador_Modelo_Fraude import preparacionDatos, preparaciónTrainyTest, generacionModeloRegression, generacionModeloBosqueRandom ,generacionModeloArbolDesicion, entrenamiento, predicción, accuracy_scoreFraude, validacion_cruzada

def main():

 print("------------------------------ Analizador contra Fraude ------------------------------ ")

 #Se paramos los datos para el entrenamiento, validacion y pruebas: 
 X, y = preparacionDatos()

 #Separamos los datos para el entrenamiento y la validación
 X_train, X_test, y_train, y_test = preparaciónTrainyTest(X, y, 0.15)
 print(f"Datos de entrenamiento X: \n {X_train}")
 print(f"\nDatos de entrenamiento y: \n {y_train}")

 #Configuramos los modelos así como los hiperparametros
 modeloClasificacion = generacionModeloRegression()
 modeloRandomForest = generacionModeloBosqueRandom(200, 10, 5, 2,'sqrt', True, 42, -1)
 ModeloArbolDesicion = generacionModeloArbolDesicion(None)

 #Usamos cross-validation para testear el modelo. 

 cross_list = validacion_cruzada(ModeloArbolDesicion, X_train, y_train, 3, 'accuracy')
 print(f"\nCross validation con 3 folds: ")
 print(f"El accuracy para cada fold es: {cross_list}")
 print(f"El promedio es de: {cross_list.mean()}")
 print(f"La sensibilidad a los datos es(varianza): {cross_list.std()}")

 cross_list = validacion_cruzada(ModeloArbolDesicion, X_train, y_train, 4, 'accuracy', 1)
 print(f"\nCross validation con 4 folds estratificado: ")
 print(f"El accuracy para cada fold es: {cross_list}")
 print(f"El promedio es de: {cross_list.mean()}")
 print(f"La sensibilidad a los datos es(varianza): {cross_list.std()}")

 cross_list = validacion_cruzada(ModeloArbolDesicion, X_train, y_train, 4, 'accuracy')
 print(f"\nCross validation con 4 folds: ")
 print(f"El accuracy para cada fold es: {cross_list}")
 print(f"El promedio es de: {cross_list.mean()}")
 print(f"La sensibilidad a los datos es(varianza): {cross_list.std()}")

 print(" ")

 cross_list = validacion_cruzada(ModeloArbolDesicion, X_train, y_train, 10, 'accuracy')
 print(f"\nCross validation con 10 folds: ")
 print(f"El accuracy para cada fold es: {cross_list}")
 print(f"El promedio es de: {cross_list.mean()}")
 print(f"La sensibilidad a los datos es(varianza): {cross_list.std()}")




main()