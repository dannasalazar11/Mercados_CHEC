import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def mostrar():
    # Definir la ruta de la carpeta donde están los modelos
    modelos_path = "Modelos"

    # Lista de modelos no lineales
    modelos_no_lineales = ["RandomForest", "GradientBoosting", "NeuralNetwork", "GaussianProcessRegressor", "GaussianProcessRegressor_Matern"]

    # Cargar solo los modelos no lineales en un diccionario
    models = {}
    for model_name in modelos_no_lineales:
        file_path = os.path.join(modelos_path, model_name + ".joblib")
        if os.path.exists(file_path):
            models[model_name] = joblib.load(file_path)

    # Lista de columnas de fecha disponibles
    fecha_columns = [
        'Fecha Actualización Estado Convocatoria', 'Inicio Periodo Contratar',
        'Fin Periodo Contratar', 'Fecha Publicacion Aviso',
        'Fecha Pliegos Para Consulta', 'Fecha Pliegos Definitivos',
        'Fecha Limite Recepción Ofertas', 'Fecha Audiencia Pública'
    ]

    # Cargar datos (suponiendo que los datos están en un archivo CSV o similar)
    df1 = pd.read_excel('Datos/df_imputado_original.xlsx')
    df1[fecha_columns] = df1[fecha_columns].apply(pd.to_datetime, errors='coerce')

    df_final = pd.read_excel('Datos/df_final.xlsx')  # Simulación de datos procesados
    feature_names = df_final.columns

    # Interfaz de usuario en Streamlit
    st.title("Visualización de Predicciones de Modelos No Lineales")

    # Selección de columna de fecha
    columna_seleccionada = st.selectbox("Seleccione la columna de fecha:", fecha_columns)

    # Definir rango de fechas
    min_date = df1[columna_seleccionada].min()
    max_date = df1[columna_seleccionada].max()
    fecha_inicio = st.date_input("Fecha Inicio", min_value=min_date, max_value=max_date, value=min_date)
    fecha_fin = st.date_input("Fecha Fin", min_value=min_date, max_value=max_date, value=max_date)

    # Selección de modelo
    modelo_seleccionado = st.selectbox("Seleccione el modelo:", list(models.keys()))

    # Botón para generar la gráfica
    if st.button("Generar Gráfica"):
        with st.spinner("Generando gráfica..."):
            # Filtrar datos en el período seleccionado
            filtered_df = df1[(df1[columna_seleccionada] >= pd.Timestamp(fecha_inicio)) & (df1[columna_seleccionada] <= pd.Timestamp(fecha_fin))]
            filtered_indices = list(set(df1.index) & set(filtered_df.index))
            filtered_indices.sort()
            
            if len(filtered_indices) == 0:
                st.warning("No hay datos en el rango de fechas seleccionado.")
            else:
                Xf = np.random.rand(len(filtered_indices), len(feature_names))  # Simulación de datos de entrada
                yf = np.random.rand(len(filtered_indices))  # Simulación de datos reales
                model = models[modelo_seleccionado]
                y_pred = model.predict(Xf)
                
                # Alinear predicciones con fechas
                y_pred_df = pd.DataFrame(y_pred, index=filtered_df.loc[filtered_indices, columna_seleccionada], columns=["Predicción"])
                
                # Graficar resultados
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(filtered_df.loc[filtered_indices, columna_seleccionada], yf, label="Real", color="blue", linestyle="dashed")
                ax.plot(y_pred_df, label="Predicción", color="red")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Valor")
                ax.set_title(f"Predicción vs Real ({modelo_seleccionado})")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Mostrar feature importances si están disponibles
                if hasattr(model, "feature_importances_"):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    importances = model.feature_importances_
                    ax.barh(feature_names, importances, color="green")
                    ax.set_xlabel("Importancia")
                    ax.set_ylabel("Características")
                    ax.set_title(f"Importancia de Características ({modelo_seleccionado})")
                    st.pyplot(fig)
