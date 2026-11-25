import pandas as pd 
import matplotlib.pyplot as plt
import math

#1. Datos a utilizar: 
x = [1, 2, 3, 4, 5]
y_real = [3, 5, 7, 9, 11]

w = 0.0
b = 0.0

epochs = 1000
epochsDos = 100

alpha_Uno = 0.0001
alpha_Dos = 0.01
alpha_Dos = 0.5
alpha_Tres = 1.0



#2. Implementación de la función MSE 

def calcular_mse(x, y_real, w, b): 
    if len(x) != len(y_real): 
      raise ValueError("La longitud de los targets no coincide con los features. ")

    suma = 0
    for i in range(len(x)): 
       suma += (((w * x[i]) + b) - y_real[i]) ** 2 
    return suma / len(x)



#3. Calculo de derivadas multivariables: 
def derivadas_multivariables(x, y_real, w, b): 
   if len(x) != len(y_real): 
     raise ValueError("La longitud de los targets no coincide con los features. ")
       
   sumaW = 0
   sumaB = 0
   for i in range(len(x)): 
    sumaW += ((((w * x[i]) + b) - y_real[i])) * x[i]
    sumaB += ((w * x[i]) + b) - (y_real[i]) 
   return (2 * sumaW) / len(x), (2 * sumaB) / len(x)

#4. Función Calculo de Gradientes: 
def calcular_gradientes(x, y_real, w, b, alpha, epochs): 
  
  if len(x) != len(y_real): 
       raise ValueError("La longitud de los targets no coincide con los features. ")
  historial_w = []
  historia_b = []
  listaMSE = []   
  for i in range(epochs):
     try:
       mse = calcular_mse(x, y_real, w, b)
       if math.isinf(mse) or math.isnan(mse):
         print(f"Divergencia (inf/nan) detectada en la iteración {i + 1}")
         break
       listaMSE.append(mse)
       gradiente = derivadas_multivariables(x,y_real, w,b)
       w = w - (alpha * gradiente[0])
       b = b - (alpha * gradiente[1])
       historial_w.append(w)
       historia_b.append(b)
     except OverflowError:
       print(f"Divergencia (Overflow) detectada en la iteración {i + 1}")
       break
  return historial_w, historia_b, listaMSE 
    

#5. Comparamos el analisis de los datos usando diferentes alpha. 

#5.1 alpha 0.0001 | iteraciones 1000
historial_w, historial_b, listaMSEUno = calcular_gradientes(x, y_real, w, b, alpha_Uno, epochs)
tablaUno = pd.DataFrame({
     "historial_w" : historial_w,
     "historial_b" : historial_b, 
     "Sucesion Errores" : listaMSEUno 
})
print(f"\nSe muestra la tabla del historial del MSE con Alpha = 0.0001 y epochs 1000: \n {tablaUno}")


#5.2 alpha 0.01 | iteraciones 1000
historial_w, historial_b, listaMSEDos = calcular_gradientes(x, y_real, w, b, 0.01, epochs)
tablaDos = pd.DataFrame({
     "historial_w" : historial_w,
     "historial_b" : historial_b, 
     "Sucesion Errores" : listaMSEDos 
})
print(f"\nSe muestra la tabla del historial del MSE con Alpha = 0.01 y epochs 1000: \n {tablaDos}")


#5.2 alpha 0.5 | iteraciones 1000
 # Al tener un alpha muy grande 0.5 el modelo diverge muy rapido haciendo que los parametros se vuelvan muy grandes. 
 # Esto hace que superen la cantidad de tamaño que python soporta, haciendo que falle el programa. 
historial_w, historial_b, listaMSETres = calcular_gradientes(x, y_real, w, b, alpha_Tres, epochs)
tablaTres  = pd.DataFrame({
     "historial_w" : historial_w,
     "historial_b" : historial_b, 
     "Sucesion Errores" : listaMSETres

})
print(f"\nSe muestra la tabla del historial del MSE con Alpha = 0.5 y epochs 1000: \n {tablaTres}")


#5.2 alpha 1 | iteraciones 100
 # Al tener un alpha muy grande 1 el modelo diverge muy rapido haciendo que los parametros se vuelvan muy grandes. 
 # Esto hace que superen la cantidad de tamaño que python soporta, haciendo que falle el programa. 
historial_w, historial_b, listaMSECuatro = calcular_gradientes(x, y_real, w, b, alpha_Tres, epochsDos)
tablaTres  = pd.DataFrame({
     "historial_w" : historial_w,
     "historial_b" : historial_b, 
     "Sucesion Errores" : listaMSECuatro

})
print(f"\nSe muestra la tabla del historial del MSE con Alpha = 1 y epochs 100: \n {tablaTres}")

#6 Graficamos la cantidad de iteraciones con el MSE


plt.figure(figsize=(8, 5))
plt.plot(range(len(listaMSEUno)), listaMSEUno, linewidth=2)
plt.title("Evolución del MSE con Alpha = 0.0001")
plt.xlabel("Número de iteraciones")
plt.ylabel("MSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("GraficasMSE/mse_iteracionesUno.png")  # Guarda la imagen si lo corres en Docker
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(len(listaMSEDos)), listaMSEDos, linewidth=2)
plt.title("Evolución del MSE con Alpha = 0.01 ")
plt.xlabel("Número de iteraciones")
plt.ylabel("MSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("GraficasMSE/mse_iteracionesDos.png")  # Guarda la imagen si lo corres en Docker
plt.show()   

plt.figure(figsize=(8, 5))
plt.plot(range(len(listaMSETres)), listaMSETres, linewidth=2)
plt.title("Evolución del MSE con Alpha = 0.5")
plt.xlabel("Número de iteraciones")
plt.ylabel("MSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("GraficasMSE/mse_iteracionesTres.png")  # Guarda la imagen si lo corres en Docker
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(len(listaMSECuatro)), listaMSECuatro, linewidth=2)
plt.title("Evolución del MSE con Alpha = 1 ")
plt.xlabel("Número de iteraciones")
plt.ylabel("MSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("GraficasMSE/mse_iteracionesCuatro.png")  # Guarda la imagen si lo corres en Docker
plt.show()   


