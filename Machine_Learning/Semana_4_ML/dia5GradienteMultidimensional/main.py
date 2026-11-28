import matplotlib.pyplot as plt

# Valores iniciales: 
theta = [3.0, -2.0, 1-0]
alpha = 0.1
epochs = 100

# 1. Función multidimensional | minimo global en (0,0,0)
def costo_MSE(theta):
    if len(theta) != 3: 
        raise ValueError("tus parametros deben tener una cantidad de 3. ")
    return pow(theta[0],2) + (2 * pow(theta[1],2)) +  (3 * pow(theta[1],2)) 

def costo_MSE_numerico(thetaUno, thetaDos, thetaTres):
    if len(theta) != 3: 
        raise ValueError("tus parametros deben tener una cantidad de 3. ")
    return pow(thetaUno,2) + (2 * pow(thetaDos,2)) +  (3 * pow(thetaTres,2)) 

#2. Calculo de las derivadas multivariables y regreso del gradiente

def derivadasMultivariables(theta):
    if len(theta) != 3: 
        raise ValueError("tus parametros deben tener una cantidad de 3. ")
    return (2 * theta[0]), (4 * theta[1]), (6 * theta[2])

#3 implementamos el Descenso de gradiente

def DescensoGradiente(theta, alpha, epochs): 
   if len(theta) != 3: 
        raise ValueError("tus parametros deben tener una cantidad de 3. ")

   evoCostos = [costo_MSE(theta)]
   thetaAux = [theta[0], theta[1], theta[2]]
   evoThetaUno = [theta[0]]
   evoThetaDos = [theta[1]]
   evoTheataTres = [theta[2]]
   for i in range(len(epochs)):
        theta[0] = theta[0] - ( alpha * derivadasMultivariables(thetaAux)[0])
        theta[1] = theta[1] - ( alpha * derivadasMultivariables(thetaAux)[1])
        theta[2] = theta[2] - ( alpha * derivadasMultivariables(thetaAux)[2])
        thetaAux = [theta[0], theta[1], theta[2]]
        evoCostos.append(costo_MSE(theta))
        evoThetaUno.append(theta[0])
        evoThetaDos.append(thetaDos[1])
        evoThetaTres.append(theta[2])
    
   return evoCostos, evoThetaUno, evoThetaDos, evoTheataTres


#4 Comparación por aproximación numérica: 
#Función multivariable: 
def derivada_numerica_multivariable(funcion, thetaUno, thetaDos, thetaTres, h=0.00001): 
    return (round(((funcion(thetaUno + h, thetaDos, thetaTres) - funcion(thetaUno - h, thetaDos, thetaTres))/ (2 * h)), 1),
           round(((funcion(thetaUno, thetaDos + h, thetaTres) - funcion(thetaUno, thetaDos - h, thetaTres))/ (2 * h)), 1),
           round(((funcion(thetaUno, thetaDos, thetaTres + h) - funcion(thetaUno, thetaDos, thetaTres - h))/ (2 * h)), 1))

print(f"Comparación del Método Analitico y el método númerico: ")
print(f"El gradiente con el método analitico es: {derivadasMultivariables(theta)}")
print(f"El gradiente con el método numérico es: {derivada_numerica_multivariable(costo_MSE_numerico, theta[0], theta[1], theta[2])}")


#5 Vizualización de los datos

evoCostos, evoThetaUno, evoThetaDos, evoTheataTres = DescensoGradiente(theta, alpha, epochs)
# Gráfica 1: Evolución del costo

plt.figure(figsize=(8, 4))
plt.plot(evoCostos, color="crimson", linewidth=2)
plt.title("Evolución de la función de costo")
plt.xlabel("Iteraciones")
plt.ylabel("Valor del costo J")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("graficasVisualizacion/evolucion_costo.png")


# Gráfica 2: Evolución de los parámetros
plt.figure(figsize=(8, 4))
plt.plot(evoThetaUno, label="θ₁", linewidth=2)
plt.plot(evoThetaDos, label="θ₂", linewidth=2)
plt.plot(evoTheataTres, label="θ₃", linewidth=2)
plt.axhline(0, color="black", linestyle=":", linewidth=1)  # Referencia en el mínimo (0)
plt.title("Evolución de los parámetros")
plt.xlabel("Iteraciones")
plt.ylabel("Valor del parámetro")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("graficasVisualizacion/evolucion_parametros.png")




