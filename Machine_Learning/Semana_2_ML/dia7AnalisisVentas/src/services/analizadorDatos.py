
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.utils.logger import obtener_logger
from pathlib import Path

raiz_proyecto = Path(__file__).resolve().parent.parent.parent
ruta_guardado = raiz_proyecto / "ResultadosGraficas" / "HistogramaVentas.png"


logger = obtener_logger(__name__)

def paraDataFrame(tabla_csv):
    logger.info("analizador Grafico | Se hizo la conversión de csv a un DataFrame.")
    return pd.read_csv(tabla_csv)

def limpiezaDatos(tablaDatos): 
   
   # ----------- Fecha --------------------------------------------------
   tablaDatos["Fecha"] = pd.to_datetime(tablaDatos["Fecha"], format="mixed", errors="coerce")

   tablaDatos["Fecha"] = tablaDatos["Fecha"].fillna(datetime.now())
   tablaDatos["Fecha"] = pd.to_datetime(tablaDatos["Fecha"])

   tablaDatos["Mes"] = tablaDatos["Fecha"].dt.month
   tablaDatos["Dia"] = tablaDatos["Fecha"].dt.day
   tablaDatos = tablaDatos.drop(columns=["Fecha"], errors="ignore")

   # ----------- Producto -----------------------------------------------

   tablaDatos["Producto"] = tablaDatos["Producto"].replace("Memoria RAM 16 GB","Memoria RAM 16GB")
   tablaDatos["Producto"] = tablaDatos["Producto"].replace("Laptop ","Laptop")
   tablaDatos["Producto"] = tablaDatos["Producto"].astype("string")

   # ----------- Ciudad -----------------------------------------------
   
   tablaDatos["Ciudad"] = tablaDatos["Ciudad"].astype(str).str.strip()
   tablaDatos["Ciudad"] = tablaDatos["Ciudad"].replace("Guadalajra", "Guadalajara")


   # ----------- Canal -----------------------------------------------
   
   tablaDatos["Canal"] = tablaDatos["Canal"].astype(str).str.strip()
   tablaDatos["Canal"] = tablaDatos["Canal"].str.capitalize()
   tablaDatos["Canal"] = tablaDatos["Canal"].replace("Tienda fisica","Tienda física")

   # ----------- Cantidad -----------------------------------------------
      
   tablaDatos["Cantidad"] =  pd.to_numeric(tablaDatos["Cantidad"], errors= "coerce")
   tablaDatos["Cantidad"] = tablaDatos["Cantidad"].abs()
   tablaDatos = tablaDatos.dropna(subset=["Cantidad"])

   print(tablaDatos[tablaDatos["Cantidad"] < 0])
   
   # ----------- Precio_Unitario -----------------------------------------------
      
   tablaDatos["Precio_Unitario"] = pd.to_numeric(tablaDatos["Precio_Unitario"], errors= "coerce")
   tablaDatos = tablaDatos.dropna(subset=["Precio_Unitario"])
   tablaDatos["Precio_Unitario"] = tablaDatos["Precio_Unitario"].abs()


   # ----------- Descuento -----------------------------------------------
      
   tablaDatos["Descuento"] = pd.to_numeric(tablaDatos["Descuento"], errors="coerce")
   tablaDatos = tablaDatos.dropna(subset=["Descuento"])
 

   
   # ----------- Ventas -----------------------------------------------
      
   tablaDatos["Ventas"] = pd.to_numeric(tablaDatos["Ventas"], errors="coerce")
   tablaDatos = tablaDatos.dropna(subset=["Ventas"])
   tablaDatos["Ventas"] = tablaDatos["Ventas"].abs()
   

   # ----------- SubTotal -----------------------------------------------
   tablaDatos["SubTotal"] = tablaDatos["Cantidad"] * tablaDatos["Precio_Unitario"]
   tablaDatos = tablaDatos.drop(columns=["Precio_Unitario", "Cantidad"], errors="ignore")

   #Convertimos a valores binarios los datos para su entrenamiento en el modelo de regressión. 
   #drop_fisrt elimina una columna ya que se vuelve redundante, si se tiene que en todas es 0 entonces es porque esta en la otra. 
   tablaDatos = pd.get_dummies(tablaDatos, columns=["Producto","Ciudad", "Canal"], drop_first=True, dummy_na=False)

   logger.info("analizador Grafico | Se limpio el dataframe de datos erroneos. ")
   return tablaDatos

def primerasFilas(tabla):
   
   print("\nPrimeras 5 filas de la tabla:")
   print(f"\n{tabla.head()}")
   logger.info("analizador Grafico | Se mostraron las primeras 5 filas. ")

def dimension(tabla): 
   filas, columnas = tabla.shape
   print(f"\nDimension de la tabla | filas: {filas} columnas: {columnas}")
   logger.info("analizador Grafico | Se mostro la dimensión de la tabla. ")

def infoTabla(tabla):
   print("\nResumen de la tabla: ")
   tabla.info()
   logger.info("analizador Grafico | Se mostro información de la tabla. ")

def resumenEstadistico(tabla): 
    print(f"\n Mostramos el resumen estadistico: ")
    print(tabla.describe())
    logger.info("analizador Grafico | Se mostro el resumen estadistico de la tabla. ")
   
def histogramaDeVentas(tabla):
   tabla["Ventas"].hist(bins=30, color='skyblue', edgecolor='black')

   plt.title("Comportamiento de las ventas")
   plt.xlabel("Monto de la Venta")
   plt.ylabel("Frecuencia de la Venta")
   plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
   plt.close()
   logger.info("analizador Grafico | Mostramos el histograma de Ventas. ")
