import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

def mostrar():
    st.title("Modelos de Predicción Clásicos Lineales")

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
    start_date = st.date_input("Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("Fecha de Fin", max_date, min_value=min_date, max_value=max_date)

    # Botón para generar la gráfica
    if st.button("Generar Gráfica"):
        plot_predictions(df1, column_selector, start_date, end_date, model_selector, models)

# Función para graficar predicciones
def plot_predictions(df1, column_selector, start_date, end_date, model_selector, models):

        for nombre in ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'train_idx', 'test_idx', 'ind']:
            globals()[nombre] = np.load(f'Datos/Arreglos/{nombre}.npy')

        df_final = pd.read_excel('Datos/df_final.xlsx')

        col = column_selector
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        model_name = model_selector
    
        if start_date and end_date and model_name:
            # Filtrar datos en el período seleccionado
            filtered_df = df1[(df1[col] >= start_date) & (df1[col] <= end_date)]
            filtered_indices = list(set(test_idx) & set(filtered_df.index))
            filtered_indices.sort()
    
            if len(filtered_indices) == 0:
                st.warning("No hay datos en el rango de fechas seleccionado.")
                return
    
            Xf = X[filtered_indices]
            yf = y[filtered_indices]
            model = models[model_name]
            y_pred = model.predict(Xf)
    
            # Alinear las predicciones con las fechas correctas
            y_pred_df = pd.DataFrame(y_pred, index=filtered_df.loc[filtered_indices, col], columns=["Predicción"])
    
            # Crear la figura con dos subgráficas (una al lado de la otra)
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(f"{model_name}", fontsize=16)
    
            # Primera gráfica: Predicciones vs serie real
            axes[0].plot(filtered_df.loc[filtered_indices, col], yf, label="Real", color="blue", linestyle="dashed")
            axes[0].plot(y_pred_df, label="Predicción", color="red")
            axes[0].set_xlabel("Fecha")
            axes[0].set_ylabel("Valor")
            axes[0].set_title("Predicción vs Real")
            axes[0].legend()
            axes[0].tick_params(axis='x', rotation=45)
    
            # Segunda gráfica: Coeficientes del modelo
            if model_name == "ElasticNet":
                coef = model[-1].coef_
            else:
                coef = model.coef_
    
            feature_names = df_final.columns if len(df_final.columns) == len(coef) else np.arange(len(coef))  # Nombres de características
    
            axes[1].barh(feature_names, coef, color="purple")
            axes[1].set_xlabel("Valor del coeficiente")
            axes[1].set_ylabel("Características")
            axes[1].set_title("Coeficientes del modelo")
    
            plt.tight_layout()
            st.pyplot(fig)  # Mostrar en Streamlit
    
        else:
            st.error("Modelo no encontrado.")
