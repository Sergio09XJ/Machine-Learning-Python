
import numpy as np
from sklearn.metrics import (accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                           confusion_matrix
)

def main():
    y_real = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    """
      Tabla de verdaderos y falsos positivos y negativos. 
      real/pred  1        |          0
         1       0,2,5,7                 3, 
         0        6                  1,4,7,9
    """
    print(f"\nImprimimos el porcentaje de las predicciones correctas(Accuracy): {accuracy_score(y_real, y_pred)}")
    print(f"\nImprimimos cuales TP son realmente TP (precision): {precision_score(y_real, y_pred)}")
    print(f"\nImprimimos Cuantos TP logro encontrar el modelo (recall): {recall_score(y_real, y_pred)}")
    print(f"\nImprimimos la relación entre precision y recall(f1_score): {f1_score(y_real, y_pred)}")
    print(f"\nImprimimos el tipo de aciertos y errores (conf_matrix): {confusion_matrix(y_real, y_pred)}")

main()