
#1. Datos a utilizar: 
x = [1, 2, 3, 4, 5]
y_real = [3, 5, 7, 9, 11]

#2. Implementación de la función MSE 

def calcular_mse(x, y_real, w, b): 
    if len(x) != len(y_real): 
      raise Exception("La longitud de los targets no coincide con los features. ")

    suma = 0
    for i in range(len(x)): 
       suma += (y_real[i] - ((w * x[i]) + b)) ** 2 
    return suma / len(x)


#3. Calculo de Gradientes: 
def calcular_gradientes(x, y_real, w, b): 
    if len(x) != len(y_real): 
      raise Exception("La longitud de los targets no coincide con los features. ")
    
    sumaW = 0
    sumaB = 0
    for i in range(len(x)): 
      sumaW += ((y_real[i] - ((w * x[i]) + b)) ** 2 ) * X[i]
      sumaB += (y_real[i] - ((w * x[i]) + b)) ** 2 
    return 2(sumaW) / len(x), 2(sumaB) / len(x)