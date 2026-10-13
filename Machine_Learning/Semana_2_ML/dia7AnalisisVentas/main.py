
from src.services.analizadorDatos import paraDataFrame, limpiezaDatos, primerasFilas, dimension, infoTabla, resumenEstadistico, histogramaDeVentas
from src.services.analizadorGrafico import realVSpredicho, graficaResiduos
from src.services.analizadorModelo import preparaciónTrainyTest, separacionDatos, generacionModelo, entrenamiento, coef_intercepto, prediccion, analisisGeneral
from pathlib import Path
import pandas as pd
import time 


raiz_proyecto = Path(__file__).resolve().parent
    
csv_path =  raiz_proyecto  / "Datos" /  "dataset_ventas_420.csv"
csv_pathDos =  raiz_proyecto  / "Datos" / "dataset_ventas_420_con_errores.csv"

def carga(Color, ColorFinal, RESET, mensaje):
    inicio = time.perf_counter()
    duracion = 3

    while time.perf_counter() - inicio < duracion :
          for i in range(1,5):
           print( Color + f"\rProcesando {'.' * i}      " + RESET, end="",flush=True)
           time.sleep(0.3)
    print(ColorFinal + f"\n{mensaje}" + RESET)
          
    
def main():

    #Creación y limpieza de la tabla
    tabla = paraDataFrame(csv_pathDos)
    tabla = limpiezaDatos(tabla)

    #Analisis de datos de la tabla 
    primerasFilas(tabla)
    infoTabla(tabla)
    dimension(tabla)
    resumenEstadistico(tabla)
    histogramaDeVentas(tabla)

    ventasAltas = tabla[tabla["Ventas"] >= 5000]

    print(f"Imprimimos las ventas altas(outliners): \n {ventasAltas}")

    #Separamos en los datos de entrenamiento(features) y el target
    X, y  = separacionDatos(tabla)

    #Separamos para features de entrenamiento y prueba así como target de entrenamiento y prueba. 
    X_train, X_test, y_train, y_test = preparaciónTrainyTest(X, y)

    #Instancia del modelo 
    modelo = generacionModelo()
    #Entrenamiento del modelo 
    modelo = entrenamiento(X_train, y_train, modelo)

    #Conocemos los valores de cambio de cada X así como el valor base cuando X es 0. 
    coef, intercepto = coef_intercepto(modelo)

    tabla_coeficientes = pd.DataFrame({
        "Variable" : X.columns , 
        "Impacto de ventas" : coef 
    })
    #Imprimimos los datos. 
    print(f"\nImprimimos el valor de y cuando las X son 0: {intercepto}")
    print(f"\nImprimimos el cambio de cada variable de X con el nombre de esta: ")
    print(f"{tabla_coeficientes}")

    #Predecimos. 
    y_sombrero = prediccion(X_test, modelo)
    print(f"\nValores de Venta predichos: {y_sombrero[:30]}")
    print(f"\nValores de Venta test: {y_test[:30]}")

    #Mostramos el analisis general. 
    analisisGeneral(y_test, y_sombrero)

    #Mostramos las graficas para conocer la exactitud del modelo. 
    realVSpredicho(y_test, y_sombrero)
    graficaResiduos(y_test, y_sombrero)
    


main()