
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score

import pandas as pd
from  pathlib import Path 

# 1. importamos el CSV 
raiz_proyecto = Path(__file__).resolve().parent
datasetFotonica = pd.read_csv(raiz_proyecto / "datos" /"dataset_fotonica.csv")
#print(datasetFotonica)

# 2. Separamos los datos en features y target después para el entrenamiento - validación - test 
y = datasetFotonica["eficiencia_acoplamiento_pct"]
X = datasetFotonica.drop(columns="eficiencia_acoplamiento_pct")
#print(f"Features: \n{X}")
#print(f"Target: \n{y}")

X_aux, X_val, y_aux, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_aux, y_aux, test_size=0.1, random_state=42)

# 2.1 Escalamos los datos haciendo uso de  StandardScaler  y Min Max

scaler = StandardScaler()
scalerMM = MinMaxScaler()
X_escalado = scaler.fit_transform(X_train)
X_minMax = scalerMM.fit_transform(X_train)


# 3. Creamos 4 modelos de Arbol de desicion, simple, normal, complejo y escalado min max normal. 
modeloKNNs  = KNeighborsRegressor(n_neighbors=19, weights="uniform", p=2)
modeloKNNm = KNeighborsRegressor(n_neighbors=10, metric="manhattan")
modeloKNNmMM = KNeighborsRegressor(n_neighbors=10, metric="manhattan")
modeloKNNg = KNeighborsRegressor(n_neighbors=2, weights="distance")

#4. Entrenamiento de modelos

modeloKNNs.fit(X_escalado, y_train)
modeloKNNm.fit(X_escalado, y_train)
modeloKNNmMM.fit(X_minMax, y_train)
modeloKNNg.fit(X_escalado, y_train)

#5.  escalado de valores para la valuación y test. 

X_val_escalado = scaler.transform(X_val)
X_val_min_max = scalerMM.transform(X_val)
X_test_escalado = scaler.transform(X_test)
X_test_min_max = scalerMM.transform(X_test)

#6. Predicción de Valores para valuación y test

r_2_Train = []
r_2_Validacion = []
r2_Test  = []


r_2_Train.append(r2_score(y_train, modeloKNNs.predict(X_escalado)))
r_2_Train.append(r2_score(y_train, modeloKNNm.predict(X_escalado)))
r_2_Train.append(r2_score(y_train, modeloKNNmMM.predict(X_minMax)))
r_2_Train.append(r2_score(y_train, modeloKNNg.predict(X_escalado)))

r_2_Validacion.append(r2_score(y_val, modeloKNNs.predict(X_val_escalado)))
r_2_Validacion.append(r2_score(y_val, modeloKNNm.predict(X_val_escalado)))
r_2_Validacion.append(r2_score(y_val, modeloKNNmMM.predict(X_val_min_max)))
r_2_Validacion.append(r2_score(y_val, modeloKNNg.predict(X_val_escalado)))

r2_Test.append(r2_score(y_test, modeloKNNs.predict(X_test_escalado)))
r2_Test.append(r2_score(y_test, modeloKNNm.predict(X_test_escalado)))
r2_Test.append(r2_score(y_test, modeloKNNmMM.predict(X_test_min_max)))
r2_Test.append(r2_score(y_test, modeloKNNg.predict(X_test_escalado)))

#7. Tabla de Accuracy para modelos con diferentes hiperparámetros

dataframeAccuracy = pd.DataFrame({
    "Modelo" :   ["Modelo Pequeño", "Modelo Mediano", "Modelo Mediano Min Max", "Modelo Grande"],
    "R_2Train" : r_2_Train,
    "R_2Validacion" : r_2_Validacion, 
    "R_2Test" : r2_Test 
})
print(f"El DataFrame de r_2 con varios tipos: \n {dataframeAccuracy}")

#8. Uso de Pipeline para el entrenamiento - validación y test del modelo 
pipe = Pipeline([
    ("scaler", MinMaxScaler()),     
    ("model", KNeighborsRegressor(n_neighbors=10, metric="manhattan")) 
])

#9 División de los datos para su comparación con el método manual 
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(X_aux, y_aux, test_size=0.1, random_state=42)

#10 Entrenamiento de los datos usando cross validation y no split
scalerCross = cross_val_score(pipe, X_train_pipe, y_train_pipe, cv=5, scoring="r2")
print(f"\nLa presición de los datos usando cross es de: {scalerCross.mean():.5f}")
print(f"\nLa desiviación estandar  de los datos usando cross es de: {scalerCross.std():.5f}")


#11. Entrenamiento y test
pipe.fit(X_train_pipe, y_train_pipe)

y_sombrero_pipe = r2_score(y_test_pipe, pipe.predict(X_test_pipe))

#12. Tabla de comparación
tablar2 = pd.DataFrame({
    "Método" : ["CROSS-PIPE", "SPLIT-MANUAL"], 
    "r^2"    : [y_sombrero_pipe, r2_score(y_test, modeloKNNmMM.predict(X_test_min_max))]
})


print(f"\nEl DataFrame de r_2 comparación manual, pipeline: \n {tablar2}")
print("\n")