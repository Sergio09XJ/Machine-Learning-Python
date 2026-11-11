
from src.services.analizador_Modelo_Fraude import preparacionDatos, preparaciónTrainyTest, generacionModeloBosqueRandom, entrenamiento, predicción, MinMaxEscalado, escaladoEstadar, generacionModeloVecinos, accuracy_scoreFraude
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
 modeloVecinosEscaladoEstandar = generacionModeloVecinos()
 modeloVecinosSNEscalar = generacionModeloVecinos()
 modeloVecinosEscaladoMaxMin = generacionModeloVecinos()

 #Instancias para el escalado 
 escaladoEstandarInst = StandardScaler()
 MinMaxEscaladoInst = MinMaxScaler()

 #Escalamos los Datos de train
 X_train_escaladoEstandar = escaladoEstadar(escaladoEstandarInst, 1, X_train)
 X_train_escaladoMinMax = MinMaxEscalado(MinMaxEscaladoInst, 1, X_train)
 np.set_printoptions(suppress=True, precision=4)
 print(" ")
 print(f"\nLa media de los datos escalados es: {X_train_escaladoEstandar.mean(axis=0)}")
 print(f"\nLa desviación de los datos escalados es: {X_train_escaladoEstandar.std(axis=0)}")
 print(" ")
 print(f"Los datos ya escalados con Min y Max son: \n{X_train_escaladoMinMax}")
 print(" ")

 #Listas para la tabla.
 listaEstandar = ["Estandar"]
 listaMinMax = ["MinMax"]
 listaSN = ["SN"]
 listaRandomForest = ["Random F."]

 #Entrenamos los modelos 
 modeloVecinosEscaladoEstandar.fit(X_train_escaladoEstandar, y_train)
 modeloVecinosEscaladoMaxMin.fit(X_train_escaladoMinMax, y_train)
 modeloVecinosSNEscalar.fit(X_train, y_train)
 modeloRandomForest.fit(X_train, y_train)

 #Escalamos los valores para la verificación y predecimos. 

 X_validacion_escaladoEstandar = escaladoEstadar(escaladoEstandarInst, 0, X_validacion)
 X_validacion_escaladoMinMax = MinMaxEscalado(MinMaxEscaladoInst, 0, X_validacion)

 listaEstandar.append(accuracy_scoreFraude(modeloVecinosEscaladoEstandar.predict(X_validacion_escaladoEstandar), y_validacion))
 listaMinMax.append(accuracy_scoreFraude(modeloVecinosEscaladoMaxMin.predict(X_validacion_escaladoMinMax), y_validacion))
 listaSN.append(accuracy_scoreFraude(modeloVecinosSNEscalar.predict(X_validacion), y_validacion))
 listaRandomForest.append(accuracy_scoreFraude(modeloRandomForest.predict(X_validacion), y_validacion))

 # Escalamos los valores para test y predecimos. 

 X_test_escaladoEstandar = escaladoEstadar(escaladoEstandarInst, 0, X_test)
 X_test_escaladoMinMax = MinMaxEscalado(MinMaxEscaladoInst, 0, X_test)
 
 listaEstandar.append(accuracy_scoreFraude(modeloVecinosEscaladoEstandar.predict(X_test_escaladoEstandar), y_test))
 listaMinMax.append(accuracy_scoreFraude(modeloVecinosEscaladoMaxMin.predict(X_test_escaladoMinMax), y_test))
 listaSN.append(accuracy_scoreFraude(modeloVecinosSNEscalar.predict(X_test), y_test))
 listaRandomForest.append(accuracy_scoreFraude(modeloRandomForest.predict(X_test), y_test))


 #Preparación de la tabla Accuracy 

 MatrizAux = [listaEstandar, listaMinMax, listaSN, listaRandomForest]
 CabeceraEscalado = ["Escalado", "Verificación","Test"]
 tabla = pd.DataFrame(MatrizAux, columns=CabeceraEscalado)
 print(tabla)


  













main()