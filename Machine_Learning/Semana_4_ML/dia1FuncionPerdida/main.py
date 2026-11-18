from math import pow
from sklearn.metrics import mean_squared_error


y_real =   [100, 200, 300, 400, 500, 600]

y_pred_A = [110, 190, 310, 390, 510, 590]

y_pred_B = [150, 180, 350, 300, 550, 500]

y_pred_A = [110, 190, 333, 390, 510, 590]

def funcPerdida(y_real, y_predicha): 
    if len(y_real) != len(y_predicha): 
        raise Exception("La longitud de tus y no coincide. ")
    suma = 0
    for i in range(0, len(y_real)): 
        suma += pow(y_real[i] - y_predicha[i], 2)

    return suma / len(y_real)

try: 
    print("\nFunción manual para MSE: ")
    print(f" El MSE de y_pred_A es: {funcPerdida(y_real, y_pred_A)}")
    print(f" El MSE de y_pred_B es: {funcPerdida(y_real, y_pred_B)}")

    print("\n Función sklearn para MSE: ")
    print(f" El MSE de y_pred_A es: {mean_squared_error(y_real, y_pred_A)}")
    print(f" El MSE de y_pred_B es: {mean_squared_error(y_real, y_pred_B)}")
except ValueError as error: 
    print (error)