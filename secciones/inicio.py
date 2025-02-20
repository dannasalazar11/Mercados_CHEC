import streamlit as st
import pandas as pd

def mostrar():
    st.title("Presentación de Bases de Datos")

    st.subheader("Convocatorias_SICEP")
    st.markdown("""
                Contiene información general de las convocatorias completas para la contratación de energía en el mercado 
                mayorista, incluyendo estado, fechas clave, cantidad de energía demandada y adjudicada, y precio.
                """)
    Convocatorias_SICEP = pd.read_excel('Datos/Convocatorias_SICEP.xlsx', na_values="-")
    st.dataframe(Convocatorias_SICEP)

    st.subheader("Productos_Adj_SICEP")
    st.markdown("""
                Desglosa cada convocatoria en productos específicos, detallando la energía contratada dentro de cada proceso, 
                permitiendo ver qué productos fueron adjudicados y cuáles no.
                """)
    Productos_Adj_SICEP = pd.read_excel('Datos/Productos_Adj_SICEP.xlsx', na_values="-")
    st.dataframe(Productos_Adj_SICEP)

    st.header("Base de Datos Cruzada")
    st.markdown("""
                Cruzamos ambas bases de datos por el **código de la convocatoria** para relacionar las convocatorias con sus 
                productos específicos y así analizar si todas las convocatorias adjudicaron su energía o si hubo productos desiertos.
                """)
    Xdata = pd.read_excel('Datos/df_imputado.xlsx')
    st.dataframe(Xdata)

    st.header("Base de Datos Final (Cruzada y Preprocesada)")
    st.markdown("""
    ✅ Se filtran solo las convocatorias adjudicadas.  
    ✅ Se eliminan columnas post-adjudicación para asegurarse de que el modelo aprenda solo a partir de información que está disponible antes de la adjudicación.  
    ✅ Se codifican variables categóricas. 
    ✅ Se normalizan los datos con MinMaxScaler.  
    ✅ Se dividen en conjuntos de entrenamiento, validación y prueba.  
    """)
    df_final = pd.read_excel('Datos/df_final.xlsx')
    st.dataframe(df_final)
