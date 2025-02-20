import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

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
