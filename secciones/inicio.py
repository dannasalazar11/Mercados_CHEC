import streamlit as st
import pandas as pd

def mostrar():
    st.title("Presentación de Bases de Datos")

    st.subheader("Visualización de la Base de Datos Convocatorias_SICEP")
    Convocatorias_SICEP = pd.read_excel('Datos/Convocatorias_SICEP.xlsx', na_values="-")
    st.dataframe(Convocatorias_SICEP)

    st.subheader("Visualización de la Base de Datos Productos_Adj_SICEP")
    Productos_Adj_SICEP = pd.read_excel('Datos/Productos_Adj_SICEP.xlsx', na_values="-")
    st.dataframe(Productos_Adj_SICEP)

    st.header("Base de Datos Cruzada")
    Xdata = pd.read_excel('Datos/df_imputado.xlsx')
    st.dataframe(Xdata)

    st.header("Base de Datos Final (Cruzada y Preprocesada)")
    df_final = pd.read_excel('Datos/df_final.xlsx')
    st.dataframe(df_final)
