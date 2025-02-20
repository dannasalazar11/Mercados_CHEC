import streamlit as st
from secciones import inicio, lineales, no_lineales, tabnet, propuesta  # Importar las secciones de la carpeta "pages"

st.sidebar.title("Índice")
sections = ["Inicio", "Regresores Clásicos Lineales", "Regresores Clásicos No Lineales", "TabNet" "Propuesta"]
choice = st.sidebar.radio("Selecciona una sección", sections)

if choice == "Inicio":
    inicio.mostrar()

elif choice == "Regresores Clásicos Lineales":
    lineales.mostrar()

elif choice == "Regresores Clásicos No Lineales":
    no_lineales.mostrar()

elif choice == "TabNet":
    tabnet.mostrar()

elif choice == "Propuesta":
    propuesta.mostrar()