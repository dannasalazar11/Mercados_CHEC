import streamlit as st
from pages import inicio, regresores  # Importar las secciones de la carpeta "pages"

st.set_page_config(page_title="Navegación por secciones", layout="wide")

st.sidebar.title("Índice")
sections = ["Inicio", "Regresores Clásicos"]
choice = st.sidebar.radio("Selecciona una sección", sections)

if choice == "Inicio":
    inicio.mostrar()

elif choice == "Regresores Clásicos":
    regresores.mostrar()