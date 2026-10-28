import numpy as np
import pandas as pd

def  crearDatosNoLineales():
  #Creación de valores  X 
  X = np.linspace(1, 500, 63)
  np.random.shuffle(X)
  X = X.reshape(7,9)
  X_final = X[:, :8]
  #Creación del ruido. 
  ruido = np.random.normal(loc=1, scale=10, size=7)

  #Aplicamos a y la función no lineal mas el ruido a y
  y_aux = X[:, 8]
  y_aux = np.log(y_aux) + ruido 
  y = y_aux.reshape(7,1) 
  matriz = np.hstack((X_final, y))
  return matriz




def creacionTabla():
  matriz = crearDatosNoLineales()
  columnas = [f"feature_{i}" for i in range(9)]
  return pd.DataFrame(matriz, columns=columnas)

def creacionTablaCSV():
   creacionTabla().to_csv("tablavalores.csv" ,index=False)
  
  
creacionTablaCSV()