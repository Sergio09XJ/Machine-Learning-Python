import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Datos = {
    "ventas": [12000.50, 45000.00, 8500.20, 23000.00, 67000.00, 15000.00, 31000.00, 9200.00, 54000.00, 18500.00],
    "gastos": [8000.00, 28000.00, 6200.00, 14500.00, 41000.00, 11000.00, 19000.00, 7100.00, 32000.00, 12000.00],
    "clientes": [150, 420, 90, 230, 610, 180, 310, 110, 500, 200],
    "empleados": [2, 6, 1, 3, 8, 2, 4, 1, 7, 3],
    "giro": ["Abarrotes", "Tecnologia", "Abarrotes", "Ropa", "Tecnologia", "Ropa", "Servicios", "Abarrotes", "Servicios", "Ropa"],
    "ganancia": [4000.50, 17000.00, 2300.20, 8500.00, 26000.00, 4000.00, 12000.00, 2100.00, 22000.00, 6500.00]
}

DatosFrame = pd.DataFrame(Datos) 

# 1. Features: ventas, gastos, clientes, empleados, giro. 
# 2. Target: ganancia
# 3. Variables númericas Continuas: Ventas, gastos, ganancia | Discretas: clientes, empleados
# Variables Categóricas: giro(puede tomar números discretos para su identifficación)

print(f"\n Imprimimos la Tabla de datos: \n {DatosFrame}")
print(f"\n Imprimimos los Tipos: \n {DatosFrame.dtypes}")
print(f"\n Imprimimos: \n {DatosFrame.describe()}")

print(f"\nPromedio de ventas: {DatosFrame["ventas"].mean()}")
print(f"\nPromedio de gastos: {DatosFrame["gastos"].mean()}")
print(f"\nNegocio mayor ganancia: {DatosFrame.loc[DatosFrame["ganancia"].idxmax()]}")
print(f"\nNegocio menor ganancia: {DatosFrame.loc[DatosFrame["ganancia"].idxmin()]}")

X = np.array([DatosFrame["ventas"], DatosFrame["gastos"], DatosFrame["clientes"], Datos["empleados"]])
print(f"\n Imprimimos los Features \n {X}")

y = np.array([DatosFrame["ganancia"]])
print(f"\n Imprimimos el Target \n {y}")


plt.scatter(DatosFrame["ventas"], DatosFrame["ganancia"], color= "red")

plt.xlabel("Ventas")
plt.ylabel("Ganancia")

plt.title("Relación Ventas | Ganancia")
plt.grid(True)
plt.show()

