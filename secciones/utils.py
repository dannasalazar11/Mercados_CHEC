import pandas as pd
import joblib
import os

def cargar_modelos(modelos_path="Modelos"):
    """Carga los modelos guardados en .joblib en un diccionario."""
    model_files = [f for f in os.listdir(modelos_path) if f.endswith(".joblib")]
    return {file.replace(".joblib", ""): joblib.load(os.path.join(modelos_path, file)) for file in model_files}

def cargar_datos():
    """Carga y preprocesa las bases de datos necesarias."""
    df = pd.read_excel('Datos/df_imputado_original.xlsx')
    return df