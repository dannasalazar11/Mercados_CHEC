import streamlit as st
import pandas as pd
import joblib
import os

def main():
    st.set_page_config(page_title="Navegación por secciones", layout="wide")
    
    st.sidebar.title("Índice")
    sections = ["Inicio", "clasicos", "tabnet", "propuesta"]
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
    
    elif choice == "clasicos":
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
        print("Modelos cargados:", list(models.keys()))
    
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
