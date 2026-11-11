
from src.services.analizador_Modelo_Fraude import preparacionDatos, preparaciónTrainyTest, generacionModeloBosqueRandom, entrenamiento, predicción, MinMaxEscalado, escaladoEstadar, generacionModeloVecinos
import numpy as np
import pandas as pd
def main():

 print("------------------------------ Analizador contra Fraude ------------------------------ ")

 #Se paramos los datos para el entrenamiento, validacion y pruebas: 

 X, y = preparacionDatos()

 #Separamos los datos para el entrenamiento y la validación
 X_aux, X_test, y_aux, y_test = preparaciónTrainyTest(X, y, 0.15)
 X_train, X_validacion, y_train, y_validacion = preparaciónTrainyTest(X_aux, y_aux, 0.15)

 #Conocemos los valores a usar en la Normalización 
 aux = np.array(["timestamp","amount","transactions_last_24h","distance_from_home_km","is_foreign_country","device_score","is_night_transaction"])
 matriz = []
 for i in range(6):
  valMax = X_train[:, i].max()
  valMin = X_train[:, i].min()
  mean = X_train[:,i].mean()
  desviacion = np.std(X_train[:,i])
  matriz.append([aux[i], valMax, valMin, mean, desviacion])

 cabecera = ["Feature", "Máximo","Mínimo","Media","Desviacion"]
 tabla = pd.DataFrame(matriz, columns=cabecera)
 print(tabla)

 #Configuramos los modelos así como los hiperparametros
 modeloRandomForest = generacionModeloBosqueRandom(200, 10, 5, 2,'sqrt', True, 42, -1)
 modeloVecinos = generacionModeloVecinos()







main()