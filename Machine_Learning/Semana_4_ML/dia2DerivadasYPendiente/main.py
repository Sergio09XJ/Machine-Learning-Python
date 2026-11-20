from math import pow

# 2da Parte implementación de las funciones en python. 
print("\nFase 2 - Derivadas númericas -------------------------")
def primerafunc(x): 
    return pow(x, 2)

def segundafunc(x): 
    return (3 * pow(x, 2)) + (2 * x) +1

def tercerafunc(x): 
    return (2 * pow(x,3)) - (4 * pow(x,2)) + 5

#Implementación de la formula general para derivar. 
def derivada_numerica(funcion, x, h=0.00001): 
    return (funcion(x + h) - funcion(x - h)) / (2 * h) 

listaX = [-2,-1,0,1,2]

listaPF = []
listaSF = []
listaTF = []

#Pruebas --------------------
for i in range(len(listaX)): 
    listaPF.append(round(derivada_numerica(primerafunc, listaX[i]), 3))
    listaSF.append(round(derivada_numerica(segundafunc, listaX[i]), 3))
    listaTF.append(round(derivada_numerica(tercerafunc, listaX[i]),3))

print(f"\n El resultado de la derivdad para x  es {listaPF}")
print(f"\n El resultado de la derivdad para x  es {listaSF}")
print(f"\n El resultado de la derivdad para x  es {listaTF}")

# 3. Implementación de la función MSE completa 

print("\nFase 3 - Implementación MSE Completa y cambio de perdida respecto a w y b ------------------------- ")
X = [1, 2, 3, 4] #Features
y = [3, 5, 7, 9] #Targets 

w = 0.5 #Parametro peso. 
b = 5.0 #Parametro sesgo

#Función MSE 
def MSEcompleta(y, X, w, b): 
    if len(y) != len(X):
        raise Exception("La longitud de target es diferente a la de los Features ")
    suma = 0
    for i in range(len(X)): 
        suma += pow((y[i] - ((w * X[i]) + b)),2)
    return suma / len(X)

#Función multivariable: 
def derivada_numerica_multivariable(funcion,y, X, w, b, h=0.00001): 
    return ((funcion(y, X, w + h, b) - funcion(y, X, w - h, b)) / (2 * h)), ((funcion(y, X, w, b + h) - funcion(y, X, w, b - h))/ (2 * h))


#Derivadas multivariables para w y b
print(f"\nLa perdida inicial es de: {MSEcompleta(y, X, w, b):.3f}")
gradiente = derivada_numerica_multivariable(MSEcompleta, y, X, w, b)
#Variables obtenidas: Al tener un valor negativo en w debemos aumentarlo para disminuir la perdida, b aumento lijeramente entonces debemos disminuirlo para disminuir la perdida.
print(f"\nEl cambio de perdida respecto a w es de: {round(gradiente[0],3)}")
print(f"\nEl cambio de perdida respecto a b es de: {round(gradiente[1],3)}")

#MSE con parametros modificados
print(f"\nLa perdida modificada(w) es de: {MSEcompleta(y, X, (w + 0.1), (b)):.3f}")
print(f"\nLa perdida modificada(b) es de: {MSEcompleta(y, X, (w), (b - 0.05)):.3f}")