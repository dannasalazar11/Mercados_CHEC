import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

def mostrar():
    st.title("📈 Modelos de Predicción Clásicos No Lineales")

    # Definir la ruta de la carpeta donde están los modelos
    modelos_path = "Modelos"

    # Lista de modelos no lineales
    modelos_no_lineales = ["RandomForest", "GradientBoosting", "NeuralNetwork", "GaussianProcessRegressor"]

    # Diccionario con descripciones de los modelos
    model_descriptions = {
        "RandomForest": "🌲 **Random Forest:** Algoritmo basado en múltiples árboles de decisión. Utiliza bagging para mejorar la precisión y reducir el sobreajuste.",
        "GradientBoosting": "🚀 **Gradient Boosting:** Modelo basado en árboles que construye secuencialmente modelos más fuertes corrigiendo los errores de los anteriores.",
        "NeuralNetwork": "🧠 **Red Neuronal:** Modelo Secuencial con capas densas, activaciones ReLU y optimizador Adam. Diseñado para capturar patrones complejos en los datos.",
        "GaussianProcessRegressor": "📈 **Gaussian Process (RBF Kernel):** Método bayesiano para la regresión que mide la relación entre datos usando el kernel Radial Basis Function (RBF).",    }

    # Cargar solo los modelos no lineales en un diccionario
    models = {}
    for model_name in modelos_no_lineales:
        file_path = os.path.join(modelos_path, model_name + ".joblib")
        if os.path.exists(file_path):  # Verificar que el archivo existe
            models[model_name] = joblib.load(file_path)

    # Selección de modelo
    model_selector = st.selectbox("🎯 Selecciona el modelo", list(models.keys()))

    # Mostrar la descripción del modelo seleccionado
    st.markdown(model_descriptions[model_selector])

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
    column_selector = st.selectbox("📅 Selecciona la columna de fecha", fecha_columns)

    # Obtener el rango de fechas para la columna seleccionada
    min_date = df1[column_selector].min()
    max_date = df1[column_selector].max()

    # Selección de rango de fechas
    start_date = st.date_input("📆 Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("📆 Fecha de Fin", max_date, min_value=min_date, max_value=max_date)

    # Cargar datos auxiliares
    for nombre in ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'train_idx', 'test_idx', 'ind']:
        globals()[nombre] = np.load(f'Datos/Arreglos/{nombre}.npy')

    df_final = pd.read_excel('Datos/df_final.xlsx')

    def plot_predictions(df1, column_selector, date_start, date_end, model_selector, models):
        col = column_selector
        start_date = pd.Timestamp(date_start)
        end_date = pd.Timestamp(date_end)
        model_name = model_selector
        
        filtered_df = df1[(df1[col] >= start_date) & (df1[col] <= end_date)]
        filtered_indices = list(set(test_idx) & set(filtered_df.index))
        filtered_df = filtered_df.loc[filtered_indices].sort_values(col)
        filtered_indices = filtered_df.index.tolist()
        
        if len(filtered_indices) == 0:
            st.warning("⚠️ No hay datos en el rango de fechas seleccionado.")
            return
        
        Xf = X[filtered_indices]
        yf = y[filtered_indices]
        model = models[model_name]
        y_pred = model.predict(Xf)
        r2 = round(r2_score(yf, y_pred), 2)

        if model_name == "GaussianProcessRegressor":
            feature_index = df_final.columns.get_loc(col)
            ind_ = np.argsort(Xf[:, feature_index]).reshape(-1)
            scaler = MinMaxScaler()
            X_test_scaled = scaler.fit_transform(Xf)
            y_mean, y_std = model.predict(X_test_scaled[ind_], return_std=True)
            
            plt.figure(figsize=(10, 5))
            plt.plot(filtered_df[col], y_mean, color="red", label="Predicción")
            plt.fill_between(filtered_df[col], y_mean - y_std, y_mean + y_std, alpha=0.1, color="gray", label="Intervalo de confianza")
            plt.plot(filtered_df[col], yf[ind_], "--b", label="Real")
            plt.xlabel("Fecha")
            plt.ylabel("Valor")
            plt.title(f"Predicción vs Real (R² = {r2})")
            plt.legend()
            st.pyplot(plt)
        
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(filtered_df[col], yf, label="Real", color="blue", linestyle="dashed")
            plt.plot(filtered_df[col], y_pred, label="Predicción", color="red")
            plt.xlabel("Fecha")
            plt.ylabel("Valor")
            plt.title(f"Predicción vs Real (R² = {r2})")
            plt.legend()
            st.pyplot(plt)
            
            if hasattr(model, "feature_importances_"):
                plt.figure(figsize=(8, 5))
                plt.bar(df_final.columns, model.feature_importances_, color="blue")
                plt.xlabel("Características")
                plt.ylabel("Importancia")
                plt.title("Importancia de Características")
                plt.xticks(rotation=90)
                st.pyplot(plt)

    if st.button("📈 Generar Gráfica"):
        plot_predictions(df1, column_selector, start_date, end_date, model_selector, models)