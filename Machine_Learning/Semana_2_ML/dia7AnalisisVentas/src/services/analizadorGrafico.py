
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logger import obtener_logger
from pathlib import Path

logger = obtener_logger(__name__)


raiz_proyecto = Path(__file__).resolve().parent.parent.parent
rutaGuardadoResiduos = raiz_proyecto / "ResultadosGraficas" / "ResiduosDiferencia.png"
rutaGuardadoRealVSpredicho = raiz_proyecto / "ResultadosGraficas" / "realVSpredicho.png"

def realVSpredicho(y_test, y_sombrero):

    plt.scatter(y_test, y_sombrero, color= "green")

    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title("Valores reales vs predicciones")

    plt.grid(True)
    plt.savefig(rutaGuardadoRealVSpredicho, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Creación de la tabla real vs predicho.")

def graficaResiduos(y_test, y_sombrero):
    y_test = np.array(y_test)
    y_sombrero = np.array(y_sombrero)
    residuos = y_test - y_sombrero

    plt.scatter(y_sombrero, residuos)

    plt.axhline(0)
    plt.xlabel("Predicción")
    plt.ylabel("Residuo")
    plt.title("Residuos")

    plt.grid(True)
    plt.savefig(rutaGuardadoResiduos, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Creación de la tabla diferencia entre y_test y _sombrero.")
