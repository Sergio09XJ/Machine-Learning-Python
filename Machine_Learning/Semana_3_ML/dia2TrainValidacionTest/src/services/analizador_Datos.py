import pandas as pd
import numpy as np
from datetime import datetime

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "Datos" / "Datos_Deteccion.csv"

def toDataFrame(): 
   return pd.read_csv(CSV_PATH)

def limpieza(tablaAlimpiar):
   tablaAlimpiar["transaction_id"] = tablaAlimpiar["transaction_id"].fillna("Sin ID")

   tablaAlimpiar["timestamp"] = tablaAlimpiar["timestamp"].fillna(datetime.now())
   tablaAlimpiar["timestamp"] = pd.to_datetime(tablaAlimpiar["timestamp"])

   tablaAlimpiar["Year"] = tablaAlimpiar["timestamp"].dt.year
   tablaAlimpiar["month"] = tablaAlimpiar["timestamp"].dt.month
   tablaAlimpiar["day"] = tablaAlimpiar["timestamp"].dt.day
   tablaAlimpiar["hour"] = tablaAlimpiar["timestamp"].dt.hour


   
   

   tablaAlimpiar["amount"] = pd.to_numeric(tablaAlimpiar["amount"], errors='coerce')
   tablaAlimpiar["amount"] = tablaAlimpiar["amount"].apply(lambda x: abs(x) if pd.notnull(x) else x) 
   tablaAlimpiar["amount"] = tablaAlimpiar["amount"].fillna(0).astype("Float64")
  

   tablaAlimpiar["transactions_last_24h"] = tablaAlimpiar["transactions_last_24h"].replace([-999], np.nan)
   tablaAlimpiar["transactions_last_24h"] = tablaAlimpiar["transactions_last_24h"].fillna(0)
   tablaAlimpiar["transactions_last_24h"] = tablaAlimpiar["transactions_last_24h"].astype("Int64")

   tablaAlimpiar["distance_from_home_km"] = tablaAlimpiar["distance_from_home_km"].replace([-999], np.nan)
   tablaAlimpiar["distance_from_home_km"] = tablaAlimpiar["distance_from_home_km"].fillna(0)
   tablaAlimpiar["distance_from_home_km"] = tablaAlimpiar["distance_from_home_km"].astype("Float64")

   tablaAlimpiar["is_foreign_country"] = tablaAlimpiar["is_foreign_country"].replace({"SI" :1, "NO":0})
   #print(f"\n Vemos que valores quedan en is foreign country: \n{tablaAlimpiar["is_foreign_country"].map(type).value_counts()}")
   tablaAlimpiar["is_foreign_country"] = tablaAlimpiar["is_foreign_country"].fillna(0).astype("Int64")

   tablaAlimpiar["is_high_risk_category"] = tablaAlimpiar["is_high_risk_category"].replace([-999], np.nan)
   tablaAlimpiar["is_high_risk_category"] = tablaAlimpiar["is_high_risk_category"].fillna(0)
   tablaAlimpiar["is_high_risk_category"] = tablaAlimpiar["is_high_risk_category"].astype("Int64")


   tablaAlimpiar["device_score"] = tablaAlimpiar["device_score"].replace([-999], np.nan)
   tablaAlimpiar["device_score"] = tablaAlimpiar["device_score"].fillna(0)
   tablaAlimpiar["device_score"] = tablaAlimpiar["device_score"].astype("Float64")

   tablaAlimpiar = tablaAlimpiar.dropna(subset=["is_fraud"])
   tablaAlimpiar["is_fraud"] = tablaAlimpiar["is_fraud"].astype("Int64")
   
   return tablaAlimpiar
   

def toCSV(tablaNueva):
   tablaNueva.to_csv("~/Datos/DatosLimpios.csv", index=False)