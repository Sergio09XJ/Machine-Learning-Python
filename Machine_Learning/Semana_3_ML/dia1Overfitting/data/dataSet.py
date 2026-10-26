import numpy as np
import pandas as pd

def  crearDatosNoLineales():
  #Creación de valores  X 
  X = np.linspace(1, 1000, 63)
  np.random.shuffle(X)
  X = X.reshape(7,9)

  #Creación del ruido. 
  ruido = np.random.normal(loc=0, scale=10, size=7)

  #Aplicamos a y la función no lineal mas el ruido a y
  y_aux = X[:, 8]
  y_aux = np.log(y_aux) 
  y = y_aux.reshape(1,7) + ruido 
  matriz = np.vstack[X, y]
  return matriz




def creacionTabla():
  matriz = crearDatosNoLineales()
  
  
  