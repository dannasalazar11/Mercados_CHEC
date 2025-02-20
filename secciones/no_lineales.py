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
        if os.path.exists(file_path):  # Verificar que el archivo existe
            models[model_name] = joblib.load(file_path)

    # Selección de modelo
    model_selector = st.selectbox("Selecciona el modelo", list(models.keys()))

    # Lista de columnas de fecha disponibles
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
    start_date = st.date_input("Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("Fecha de Fin", max_date, min_value=min_date, max_value=max_date)

    def plot_predictions(df1, column_selector, start_date, end_date, model_selector, models):

        for nombre in ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'train_idx', 'test_idx', 'ind']:
            globals()[nombre] = np.load(f'Datos/Arreglos/{nombre}.npy')

        df_final = pd.read_excel('Datos/df_final.xlsx')

        col = column_selector
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        model_name = model_selector
    
        # Filtrar datos en el período seleccionado
        filtered_df = df1[(df1[col] >= pd.Timestamp(start_date)) & (df1[col] <= pd.Timestamp(end_date))]
        # filtered_indices = filtered_df.index

        filtered_indices = list(set(test_idx) & set(filtered_df.index))
        filtered_indices.sort()

        # print(len(test_idx), filtered_df.index, filtered_indices)

        if len(filtered_indices) == 0:
            st.warning("No hay datos en el rango de fechas seleccionado.")
            return

        Xf = X[filtered_indices]
        yf = y[filtered_indices]
        model = models[model_name]
        y_pred = model.predict(Xf)

        # **Alinear las predicciones con las fechas correctas**
        y_pred_df = pd.DataFrame(y_pred, index=filtered_df.loc[filtered_indices, col], columns=["Predicción"])

        # Crear la figura con subgráficas
        if model_name == "NeuralNetwork":
            # Solo mostrar la predicción vs. la real
            plt.figure(figsize=(10, 5))
            plt.plot(filtered_df.loc[filtered_indices, col], yf, label="Real", color="blue", linestyle="dashed")
            plt.plot(y_pred_df, label="Predicción", color="red")
            plt.xlabel("Fecha")
            plt.ylabel("Valor")
            plt.title(f"Predicción vs Real ({model_name})")
            plt.legend()
            plt.xticks(rotation=45)
            st.pyplot(plt)

    # Botón para generar la gráfica
    if st.button("Generar Gráfica"):
        plot_predictions(df1, column_selector, start_date, end_date, model_selector, models)