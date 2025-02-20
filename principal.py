import streamlit as st
import pandas as pd

def main():
    st.set_page_config(page_title="Navegación por secciones", layout="wide")
    
    st.sidebar.title("Índice")
    sections = ["Inicio", "Acerca de", "Datos", "Contacto"]
    choice = st.sidebar.radio("Selecciona una sección", sections)
    
    if choice == "Inicio":
        st.title("Presentación de Bases de Datos")

        st.header("Bases de Datos Iniciales")
        
        Convocatorias_SICEP = pd.read_excel('Datos/Convocatorias_SICEP.xlsx',na_values="-")
        Productos_Adj_SICEP = pd.read_excel('Datos/Productos_Adj_SICEP.xlsx',na_values="-")

        st.subheader("Visualización de la Base de Datos Convocatorias_SICEP")
        st.dataframe(Convocatorias_SICEP)
        st.subheader("Visualización de la Base de Datos Productos_Adj_SICEP")
        st.dataframe(Productos_Adj_SICEP)

        st.header("Base de Datos Preprocesada")

        Xdata = pd.read_excel('Datos/df_imputado.xlsx')
        st.dataframe(Xdata)
        
    
    elif choice == "Acerca de":
        st.title("Acerca de")
        st.write("Aquí puedes encontrar información sobre esta aplicación y su propósito.")
    
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
