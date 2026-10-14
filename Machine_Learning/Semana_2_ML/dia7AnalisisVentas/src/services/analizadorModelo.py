from sklearn.model_selection import train_test_split #Divide X y "y" para ejecución y test 
from sklearn.linear_model import LinearRegression #Modelo de Regression
#nos permite hacer un analisis del modelo 
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error)
from math import sqrt



from src.utils.logger import obtener_logger
import numpy as np

logger = obtener_logger(__name__)

def separacionDatos(tablaDatos):
    X = tablaDatos.drop(columns=["Ventas"])
    y = np.array(tablaDatos["Ventas"])
    logger.info("Separación del target y los features")
    return X, y


def preparaciónTrainyTest(X, y):
    #X_train, X_test, y_train, y_test
    logger.info("Separación de datos para el entrenamiento y la prueba.")
    return train_test_split(X, y, test_size=0.4, random_state=42)

def generacionModelo():
    logger.info("Creación de la instancia del modelo de regresion")
    return LinearRegression()

def entrenamiento(X_train, y_train, modelo):
    logger.info("Se entreno el modelo")
    return modelo.fit(X_train, y_train)

def coef_intercepto(modelo):
    logger.info("Se retorno el valor de cambio entre cada X(coef) y el valor cuando x vale 0(intercept).")
    np.set_printoptions(suppress=True, precision=3)
    coef = np.round(modelo.coef_, decimals=3)
    intercept = np.round(modelo.intercept_, decimals=3)
    return coef, intercept

def prediccion(X_test, modelo):
    logger.info("Se hizo la predicción del modelo")
    return modelo.predict(X_test)

def r2_scoreAnalisis(y_test, y_sombrero):
    logger.info("Se  el analisis R2")
    return (r2_score(y_test, y_sombrero) * 100)

def mean_squared_errorAnalisis(y_test, y_sombrero):
    logger.info("Se hizo el MSE")
    return mean_squared_error(y_test, y_sombrero)

def mean_absolute_errorAnalisis(y_test, y_sombrero):
    logger.info("Se hizo el MAE")
    return mean_absolute_error(y_test, y_sombrero)

def analisisGeneral(y_test, y_sombrero):
    print("------------------------------")
    print("    Evaluación del modelo     ")
    print("------------------------------")
    print(f"MAE:{round(mean_absolute_errorAnalisis(y_test, y_sombrero), 3)}")
    print(f"MSE:{round(mean_squared_errorAnalisis(y_test, y_sombrero), 3)}")
    print(f"RMSE:{round(sqrt(mean_squared_errorAnalisis(y_test, y_sombrero)), 3)}")
    print(f"R²:{round(r2_scoreAnalisis(y_test, y_sombrero),3)}")
    logger.info("Se hizo la tabla de errores ")









