import numpy as np
import pandas as pd

def  crearDatosNoLineales():
  #Creación de valores  X 
  X = np.linspace(1, 500, 2000)
  np.random.shuffle(X)
  X = X.reshape(500,4)
  X_final = X[:, :3]
  #Creación del ruido. 
  ruido = np.random.normal(loc=1, scale=10, size=500)

  #Aplicamos a "y" la función no lineal mas el ruido a y
  y_aux = X[:, 3]
  y_aux = np.log(y_aux) + ruido 
  y = y_aux.reshape(500,1) 
  matriz = np.hstack((X_final, y))
  return matriz




def creacionTabla():
  matriz = crearDatosNoLineales()
  columnas = [f"feature_{i}" for i in range(4)]
  return pd.DataFrame(matriz, columns=columnas)

def creacionTablaCSV():
   creacionTabla().to_csv("tablavalores.csv" ,index=False)
  
  
creacionTablaCSV()