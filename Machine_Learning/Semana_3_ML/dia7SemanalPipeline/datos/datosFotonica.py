import numpy as np
import pandas as pd

def generar_dataset_fotonica(n_muestras=850, semilla=42):
    rng = np.random.default_rng(semilla)

    # 1. Generación de características con escalas físicas muy distintas
    longitud_onda_nm = rng.uniform(1520.0, 1580.0, n_muestras)           # Escala: ~10^3
    potencia_bombeo_mw = rng.uniform(5.0, 450.0, n_muestras)             # Escala: ~10^2
    ancho_guia_onda_um = rng.uniform(0.45, 1.85, n_muestras)             # Escala: ~10^0
    temperatura_sustrato_k = rng.uniform(285.0, 340.0, n_muestras)       # Escala: ~10^2
    concentracion_dopaje_cm3 = rng.uniform(1.0e18, 9.5e18, n_muestras)   # Escala: ~10^18
    angulo_incidencia_deg = rng.uniform(6.0, 14.0, n_muestras)           # Escala: ~10^1
    rugosidad_superficial_nm = rng.uniform(0.5, 8.0, n_muestras)         # Escala: ~10^0
    longitud_cavidad_mm = rng.uniform(1.0, 15.0, n_muestras)             # Escala: ~10^1

    # 2. Relación física no lineal (Resonancia, dispersión, deriva térmica y saturación)
    # Pico de resonancia óptica acoplado entre longitud de onda y ancho de guía de onda
    desfase_espectral = (longitud_onda_nm - 1550.0) / 18.0
    desfase_geometrico = (ancho_guia_onda_um - 1.15) / 0.35
    factor_resonancia = 38.0 * np.exp(-0.5 * (desfase_espectral**2 + desfase_geometrico**2))

    # Ganancia por bombeo y dopaje de Erbio (con saturación no lineal)
    dopaje_norm = concentracion_dopaje_cm3 / 1e18
    ganancia_activa = 14.5 * np.log1p(potencia_bombeo_mw / 25.0) * (dopaje_norm / (dopaje_norm + 3.2))

    # Penalización cuadrática por desalineación del ángulo de fibra
    perdida_angular = 1.45 * ((angulo_incidencia_deg - 10.2) ** 2)

    # Pérdidas por dispersión de Rayleigh (rugosidad) y deriva térmica en la cavidad
    perdida_scattering = 0.35 * (rugosidad_superficial_nm ** 1.45) * np.sqrt(longitud_cavidad_mm)
    deriva_termica = 0.08 * np.abs(temperatura_sustrato_k - 300.0) * (longitud_cavidad_mm / 5.0)

    # Variable objetivo (y): Eficiencia de acoplamiento óptico (%)
    ruido_experimental = rng.normal(0, 2.4, n_muestras)
    eficiencia_acoplamiento_pct = (
        28.0
        + factor_resonancia
        + ganancia_activa
        - perdida_angular
        - perdida_scattering
        - deriva_termica
        + ruido_experimental
    )

    # Acotar físicamente entre 5% y 98%
    eficiencia_acoplamiento_pct = np.clip(eficiencia_acoplamiento_pct, 5.0, 98.0)

    df = pd.DataFrame({
        "longitud_onda_nm": np.round(longitud_onda_nm, 3),
        "potencia_bombeo_mw": np.round(potencia_bombeo_mw, 2),
        "ancho_guia_onda_um": np.round(ancho_guia_onda_um, 4),
        "temperatura_sustrato_k": np.round(temperatura_sustrato_k, 2),
        "concentracion_dopaje_cm3": concentracion_dopaje_cm3,
        "angulo_incidencia_deg": np.round(angulo_incidencia_deg, 3),
        "rugosidad_superficial_nm": np.round(rugosidad_superficial_nm, 3),
        "longitud_cavidad_mm": np.round(longitud_cavidad_mm, 3),
        "eficiencia_acoplamiento_pct": np.round(eficiencia_acoplamiento_pct, 3),
    })

    return df

if __name__ == "__main__":
    df_fotonica = generar_dataset_fotonica(n_muestras=850, semilla=42)
    df_fotonica.to_csv("dataset_fotonica.csv", index=False)
    print("Archivo 'dataset_fotonica.csv' generado con éxito.")
    print(f"Dimensiones: {df_fotonica.shape}")
    print(df_fotonica.head())
