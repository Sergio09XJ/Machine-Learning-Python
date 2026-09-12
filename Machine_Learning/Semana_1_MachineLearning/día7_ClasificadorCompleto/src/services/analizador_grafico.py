
import matplotlib.pyplot as plt


def prediccionDatos(y_test, y_sobrero): 

  plt.scatter(y_test, y_sobrero, color= "red")
  plt.title("Efectividad entre los valores de test con la clasificcación del modelo")
  plt.grid(True)
  plt.show()

