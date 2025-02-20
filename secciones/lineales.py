import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def mostrar():
    st.title("📈 Modelos de Predicción Clásicos Lineales")
    st.write("---")  # Línea divisoria

    # 📌 Cargar modelos desde la carpeta "Modelos"
    modelos_path = "Modelos"
    model_files = [f for f in os.listdir(modelos_path) if f.endswith(".joblib")]
    models = {file.replace(".joblib", ""): joblib.load(os.path.join(modelos_path, file)) for file in model_files}

    # 📌 Diccionario de descripciones de modelos
    model_descriptions = {
        "LinearRegression": "📏 **Regresión Lineal:** Modelo simple que ajusta una línea recta para minimizar la diferencia entre los valores reales y predichos.",
        "Ridge": "🏋️‍♂️ **Ridge Regression:** Variante de la regresión lineal con regularización L2 para evitar el sobreajuste.",
        "Lasso": "🔪 **Lasso Regression:** Regresión lineal con regularización L1, capaz de reducir coeficientes a cero y realizar selección de características.",
        "ElasticNet": "🔗 **Elastic Net:** Combinación de Ridge y Lasso que permite un equilibrio entre regularización L1 y L2.",
    }

    # 📌 Selección de modelo en la barra lateral
    st.sidebar.header("⚙️ Configuración")
    model_selector = st.sidebar.selectbox("🎯 Selecciona el modelo", list(models.keys()))

    # Mostrar la descripción del modelo seleccionado
    st.info(model_descriptions.get(model_selector, "Modelo sin descripción disponible."))

    # 📌 Cargar datos
    fecha_columns = [
        'Fecha Actualización Estado Convocatoria', 'Inicio Periodo Contratar',
        'Fin Periodo Contratar', 'Fecha Publicacion Aviso',
        'Fecha Pliegos Para Consulta', 'Fecha Pliegos Definitivos',
        'Fecha Limite Recepción Ofertas', 'Fecha Audiencia Pública'
    ]

    df1 = pd.read_excel('Datos/df_imputado_original.xlsx')
    df1[fecha_columns] = df1[fecha_columns].apply(pd.to_datetime, errors='coerce')

    # 📌 Selección de columna de fecha
    column_selector = st.sidebar.selectbox("📅 Selecciona la columna de fecha", fecha_columns)

    # 📌 Obtener el rango de fechas para la columna seleccionada
    min_date = df1[column_selector].min()
    max_date = df1[column_selector].max()

    # 📌 Selección de rango de fechas
    start_date = st.sidebar.date_input("📆 Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("📆 Fecha de Fin", max_date, min_value=min_date, max_value=max_date)

    # 📌 Botón para generar la gráfica
    if st.sidebar.button("📈 Generar Gráfica"):
        plot_predictions(df1, column_selector, start_date, end_date, model_selector, models)

# 📊 Función para graficar predicciones
def plot_predictions(df1, column_selector, start_date, end_date, model_selector, models):

    # 📌 Cargar datos de entrenamiento y prueba
    for nombre in ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'train_idx', 'test_idx', 'ind']:
        globals()[nombre] = np.load(f'Datos/Arreglos/{nombre}.npy')

    df_final = pd.read_excel('Datos/df_final.xlsx')

    col = column_selector
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    model_name = model_selector

    if start_date and end_date and model_name:
        # 📌 Filtrar datos en el período seleccionado
        filtered_df = df1[(df1[col] >= start_date) & (df1[col] <= end_date)]
        filtered_indices = list(set(test_idx) & set(filtered_df.index))
        filtered_indices.sort()

        if len(filtered_indices) == 0:
            st.warning("⚠️ No hay datos en el rango de fechas seleccionado.")
            return

        Xf = X[filtered_indices]
        yf = y[filtered_indices]
        model = models[model_name]
        y_pred = model.predict(Xf)

        # 📌 Alinear las predicciones con las fechas correctas
        y_pred_df = pd.DataFrame(y_pred, index=filtered_df.loc[filtered_indices, col], columns=["Predicción"])

        # 📊 **Gráfico 1: Predicción vs Real**
        plt.figure(figsize=(10, 5))
        plt.plot(filtered_df.loc[filtered_indices, col], yf, label="Real", color="blue", linestyle="dashed")
        plt.plot(filtered_df.loc[filtered_indices, col], y_pred, label="Predicción", color="red")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.title(f"📊 Predicción vs Real ({model_name})")
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # 📊 **Gráfico 2: Coeficientes del Modelo**
        if hasattr(model, "coef_"):
            coef = model.coef_
            feature_names = df_final.columns if len(df_final.columns) == len(coef) else np.arange(len(coef))

            plt.figure(figsize=(10, 5))
            plt.barh(feature_names, coef, color="purple")
            plt.xlabel("Valor del coeficiente")
            plt.ylabel("Características")
            plt.title(f"🟣 Importancia de Características ({model_name})")
            st.pyplot(plt)

    else:
        st.error("⚠️ Modelo no encontrado o datos incorrectos.")

