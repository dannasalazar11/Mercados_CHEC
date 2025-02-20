import streamlit as st
from secciones import inicio, lineales, propuesta  # Importar las secciones de la carpeta "pages"

st.sidebar.title("Índice")
sections = ["Inicio", "Regresores Clásicos", "Propuesta"]
choice = st.sidebar.radio("Selecciona una sección", sections)

if choice == "Inicio":
    inicio.mostrar()

elif choice == "Regresores Clásicos":
    lineales.mostrar()

elif choice == "Propuesta":
    propuesta.mostrar()