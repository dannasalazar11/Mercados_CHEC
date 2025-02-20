import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt


def regresores_clasicos():
    st.title("Modelos de Predicción Clásicos")
    st.write("Aquí puedes encontrar información sobre esta aplicación y su propósito.")

    # Definir la ruta de la carpeta donde están los modelos
    modelos_path = "Modelos"
    
    # Obtener todos los archivos .joblib en la carpeta
    model_files = [f for f in os.listdir(modelos_path) if f.endswith(".joblib")]
    
    # Cargar todos los modelos en un diccionario
    models = {}
    for file in model_files:
        file_path = os.path.join(modelos_path, file)
        model_name = file.replace(".joblib", "")  # Nombre del modelo sin extensión
        models[model_name] = joblib.load(file_path)
    
    # Mostrar los modelos cargados
    st.write(f"Modelos cargados: {list(models.keys())}")

    st.header("Regresores Clásicos Lineales")

    # Lista de modelos lineales sin SVR
    modelos_lineales = ["ElasticNet", "Lasso"]

    fecha_columns = [
    'Fecha Actualización Estado Convocatoria', 'Inicio Periodo Contratar',
    'Fin Periodo Contratar', 'Fecha Publicacion Aviso',
    'Fecha Pliegos Para Consulta', 'Fecha Pliegos Definitivos',
    'Fecha Limite Recepción Ofertas', 'Fecha Audiencia Pública'
    ]

    for nombre in ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'train_idx', 'test_idx', 'ind']:
        globals()[nombre] = np.load(f'Datos/Arreglos/{nombre}.npy')

    
    df1=pd.read_excel('Datos/df_imputado_original.xlsx')
    df1 = df1[df1["Estado Convocatoria"] != "Cancelada"]
    df1 = df1[df1["Estado Convocatoria"] != "Cerrada y desierta"]
    df1 = df1[df1["Estado Convocatoria"] != "Abierta"]
    df1=df1.iloc[ind]
    df1.reset_index(drop=True, inplace=True)

    
    # Asegurarse de que las columnas de fecha están en formato datetime
    df1[fecha_columns] = df1[fecha_columns].apply(pd.to_datetime, errors='coerce')
    
    # Sidebar para selección de parámetros
    st.title("Parámetros de Selección")
    
    # Selección de columna de fecha
    column_selector = st.selectbox("Selecciona la columna de fecha", fecha_columns)
    
    # Obtener el rango de fechas para la columna seleccionada
    min_date = df1[column_selector].min()
    max_date = df1[column_selector].max()

    st.write(f"{min_date}{max_date}")
    
    # Selección de rango de fechas
    # start_date = st.date_input("Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    # end_date = st.date_input("Fecha de Fin", max_date, min_value=min_date, max_value=max_date)

    start_date = st.date_input("Fecha de Inicio", min_date.date(), min_value=min_date.date(), max_value=max_date.date())
    end_date = st.date_input("Fecha de Fin", max_date.date(), min_value=min_date.date(), max_value=max_date.date())

    
    # Selección de modelo
    model_selector = st.selectbox("Selecciona el modelo", list(models.keys()))
    
    # Botón para generar la gráfica
    generate_button = st.button("Generar Gráfica")
    
    # Función para graficar predicciones vs serie real con filtrado por fechas
    def plot_predictions(column_selector, start_date, end_date, model_selector):
        col = column_selector
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        model_name = model_selector
    
        if start_date and end_date and model_name in models:
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
    
    # Ejecutar la función si se presiona el botón
    if generate_button:
        plot_predictions(column_selector, start_date, end_date, model_selector)

def main():
    st.set_page_config(page_title="Navegación por secciones", layout="wide")
    
    st.sidebar.title("Índice")
    sections = ["Inicio", "Regresores Clásicos", "tabnet", "propuesta"]
    choice = st.sidebar.radio("Selecciona una sección", sections)
    
    if choice == "Inicio":
        st.title("Presentación de Bases de Datos")

        # st.header("Bases de Datos Iniciales")
        
        Convocatorias_SICEP = pd.read_excel('Datos/Convocatorias_SICEP.xlsx',na_values="-")
        Productos_Adj_SICEP = pd.read_excel('Datos/Productos_Adj_SICEP.xlsx',na_values="-")

        st.subheader("Visualización de la Base de Datos Convocatorias_SICEP")
        st.dataframe(Convocatorias_SICEP)
        st.subheader("Visualización de la Base de Datos Productos_Adj_SICEP")
        st.dataframe(Productos_Adj_SICEP)

        st.header("Base de Datos Cruzada")

        Xdata = pd.read_excel('Datos/df_imputado.xlsx')
        st.dataframe(Xdata)

        st.header("Base de Datos Final (Cruzada y Preprocesada)")

        Xdata = pd.read_excel('Datos/df_final.xlsx')
        st.dataframe(Xdata)
    
    elif choice == "Regresores Clásicos":
        regresores_clasicos()
        
    
    elif choice == "Datos":
        st.title("Visualización de Datos")
        st.write("En esta sección se mostrarían gráficos y análisis de datos.")
        st.line_chart({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    
    elif choice == "Contacto":
        st.title("Contacto")
        st.write("Si deseas ponerte en contacto, deja un mensaje aquí.")
        name = st.text_input("Nombre")
        message = st.text_area("Mensaje")
        if st.button("Enviar"):
            st.success(f"Gracias {name}, hemos recibido tu mensaje.")

if __name__ == "__main__":
    main()
