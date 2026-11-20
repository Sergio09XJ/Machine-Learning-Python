from math import pow


def primerafunc(x): 
    return pow(x, 2)

def segundafunc(x): 
    return (3 * pow(x, 2)) + (2 * x) +1

def tercerafunc(x): 
    return (2 * pow(x,3)) - (4 * pow(x,2)) + 5

def derivada_numerica(funcion, x, h=0.00001): 
    return (funcion(x + h) - funcion(x - h)) / (2 * h) 

listaX = [-2,-1,0,1,2]

listaPF = []
listaSF = []
listaTF = []

for i in range(len(listaX)): 
    listaPF.append(round(derivada_numerica(primerafunc, listaX[i]), 3))
    listaSF.append(round(derivada_numerica(segundafunc, listaX[i]), 3))
    listaTF.append(round(derivada_numerica(tercerafunc, listaX[i]),3))

print(f"\n El resultado de la derivdad para x  es {listaPF}")
print(f"\n El resultado de la derivdad para x  es {listaSF}")
print(f"\n El resultado de la derivdad para x  es {listaTF}")