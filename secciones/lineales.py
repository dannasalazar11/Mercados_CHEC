import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

def mostrar():
    st.title("Modelos de Predicción Clásicos")

    # Cargar modelos desde la carpeta "Modelos"
    modelos_path = "Modelos"
    model_files = [f for f in os.listdir(modelos_path) if f.endswith(".joblib")]
    models = {file.replace(".joblib", ""): joblib.load(os.path.join(modelos_path, file)) for file in model_files}
    
    st.write(f"Modelos cargados: {list(models.keys())}")

    # Selección de modelo
    model_selector = st.selectbox("Selecciona el modelo", list(models.keys()))

    # Cargar datos
    fecha_columns = [
        'Fecha Actualización Estado Convocatoria', 'Inicio Periodo Contratar',
        'Fin Periodo Contratar', 'Fecha Publicacion Aviso',
        'Fecha Pliegos Para Consulta', 'Fecha Pliegos Definitivos',
        'Fecha Limite Recepción Ofertas', 'Fecha Audiencia Pública'
    ]
    
    df1 = pd.read_excel('Datos/df_imputado_original.xlsx')
    df1[fecha_columns] = df1[fecha_columns].apply(pd.to_datetime, errors='coerce')

    # Selección de columna de fecha
    column_selector = st.selectbox("Selecciona la columna de fecha", fecha_columns)

    # Obtener el rango de fechas para la columna seleccionada
    min_date = df1[column_selector].min()
    max_date = df1[column_selector].max()

    # Selección de rango de fechas
    start_date = st.sidebar.date_input("Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("Fecha de Fin", max_date, min_value=min_date, max_value=max_date)

    # Botón para generar la gráfica
    if st.button("Generar Gráfica"):
        plot_predictions(df1, column_selector, start_date, end_date, model_selector, models)

# Función para graficar predicciones
def plot_predictions(df1, column_selector, start_date, end_date, model_selector, models):
    st.write(f"Generando predicción con el modelo {model_selector}...")

    # Filtrar datos
    filtered_df = df1[(df1[column_selector] >= pd.Timestamp(start_date)) & (df1[column_selector] <= pd.Timestamp(end_date))]

    if filtered_df.empty:
        st.warning("No hay datos en el rango de fechas seleccionado.")
        return

    # Generar gráfica (Ejemplo simple)
    fig, ax = plt.subplots()
    ax.plot(filtered_df[column_selector], np.random.randn(len(filtered_df)), label="Predicción", color="red")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.set_title("Predicción vs Real")
    ax.legend()
    st.pyplot(fig)
