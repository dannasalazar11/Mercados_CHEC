import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

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

    with open("Modelos/scaler_value.pkl", 'rb') as f:
        scaler = pickle.load(f)

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
        y_pred_df = pd.DataFrame(y_pred, index=filtered_df.loc[filtered_indices, col], columns=["Predicción"])


        if model_name == "GaussianProcessRegressor":
            feature_index = df_final.columns.get_loc(col)
            ind_ = np.argsort(Xf[:, feature_index]).reshape(-1)

            scaler2 = MinMaxScaler(feature_range=(0, 1))
            X_train_scaled = scaler2.fit_transform(X_train)
            X_test_scaled = scaler2.transform(Xf)

            y_mean, y_std = model.predict(X_test_scaled[ind_], return_std=True)  # Predicted output from GPR
            
            y_mean_real = scaler.inverse_transform(y_mean.reshape(-1,1)).reshape(-1)
            y_std_real = scaler.inverse_transform(y_std.reshape(-1,1)).reshape(-1)
            yf_real = scaler.inverse_transform(yf).reshape(-1)
            r2=round(r2_score(yf,y_mean),2)


            x_feature = filtered_df.loc[filtered_indices, col]  # Choose a feature for plotting

            plt.plot(x_feature, y_mean_real, color="red", label="Predicción")

            plt.fill_between(
                x_feature,
                y_mean_real - 1 * y_std_real,
                y_mean_real + 1 * y_std_real,
                alpha=0.08,
                color="black",
                label=r"Intervalo de confianza",
            )
            plt.plot(x_feature, yf_real[ind_], "--b", label="Real") #target ytest
            plt.xlabel("Fecha")
            plt.ylabel("Valor (COP/kWh)")
            plt.title(f"Predicción vs Real (R2 = {np.round(r2,1)})")
            plt.legend()
            st.pyplot(plt)
        
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(filtered_df.loc[filtered_indices, col], scaler.inverse_transform(y_pred_df.to_numpy().reshape(-1,1)), label="Predicción", color="red")
            plt.plot(filtered_df.loc[filtered_indices, col], scaler.inverse_transform(yf.reshape(-1,1)), label="Real", color="blue", linestyle="dashed")
            plt.xlabel("Fecha")
            plt.ylabel("Valor (COP/kWh)")
            plt.title(f"Predicción vs Real (R2 = {r2})")
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