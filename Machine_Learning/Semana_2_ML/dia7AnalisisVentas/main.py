
from src.services.analizadorDatos import paraDataFrame, limpiezaDatos, primerasFilas, dimension, infoTabla, resumenEstadistico, histogramaDeVentas
from src.services.analizadorGrafico import realVSpredicho, graficaResiduos
from src.services.analizadorModelo import preparaciónTrainyTest, separacionDatos, generacionModelo, entrenamiento, coef_intercepto, prediccion, analisisGeneral
from pathlib import Path
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

    #Separamos en los datos de entrenamiento(features) y el target
    X, y  = separacionDatos(tabla)

    #Separamos para features de entrenamiento y prueba así como target de entrenamiento y prueba. 
    X_train, y_train, X_test, y_test = preparaciónTrainyTest(X, y)

    #Instancia del modelo 
    modelo = generacionModelo()
    #Entrenamiento del modelo 
    modelo = entrenamiento(X_train, y_train, modelo)

    """
    coef, intecepto = coef_intercepto(modelo)

    print(f"\nImprimimos el valor de y cuando las X son 0: {intecepto}")
    print(f"\nImprimimos la taza de cambio por cada valor X: {coef}")

    y_sombrero = prediccion(X_test)

    analisisGeneral(y_test, y_sombrero)
    """


main()