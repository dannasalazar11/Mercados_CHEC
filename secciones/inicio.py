import streamlit as st
import pandas as pd

# # Configurar el layout de la página
# st.set_page_config(page_title="Análisis de Datos - SICEP", layout="wide")

def mostrar():
    # Título principal con ícono
    st.markdown("<h1 style='text-align: center;'>📊 Presentación de Bases de Datos</h1>", unsafe_allow_html=True)
    st.write("---")  # Línea divisoria

    ## 📌 SECCIÓN: Convocatorias_SICEP
    with st.container():
        st.subheader("📌 Convocatorias SICEP")
        st.markdown("""
        Contiene información general de las convocatorias completas para la contratación de energía en el mercado mayorista, 
        incluyendo estado, fechas clave, cantidad de energía demandada y adjudicada, y precio.
        """)

        # Cargar y mostrar la base de datos
        Convocatorias_SICEP = pd.read_excel('Datos/Convocatorias_SICEP.xlsx', na_values="-")

        with st.expander("📂 Ver datos de Convocatorias SICEP"):
            st.dataframe(Convocatorias_SICEP.head(10))  # Muestra solo las primeras 10 filas

        # Botón de descarga
        st.download_button("⬇️ Descargar CSV", Convocatorias_SICEP.to_csv(index=False), "Convocatorias_SICEP.csv", "text/csv")

    st.divider()  # Separador

    ## 📌 SECCIÓN: Productos_Adj_SICEP
    with st.container():
        st.subheader("📌 Productos Adjudicados SICEP")
        st.markdown("""
        Desglosa cada convocatoria en productos específicos, detallando la energía contratada dentro de cada proceso, 
        permitiendo ver qué productos fueron adjudicados y cuáles no.
        """)

        # Cargar y mostrar la base de datos
        Productos_Adj_SICEP = pd.read_excel('Datos/Productos_Adj_SICEP.xlsx', na_values="-")

        with st.expander("📂 Ver datos de Productos Adj SICEP"):
            st.dataframe(Productos_Adj_SICEP.head(10))  # Muestra solo las primeras 10 filas

        # Botón de descarga
        st.download_button("⬇️ Descargar CSV", Productos_Adj_SICEP.to_csv(index=False), "Productos_Adj_SICEP.csv", "text/csv")

    st.divider()

    ## 🔄 SECCIÓN: Base de Datos Cruzada
    with st.container():
        st.header("🔄 Base de Datos Cruzada")
        st.markdown("""
        Cruzamos ambas bases de datos por el **código de la convocatoria** para relacionar las convocatorias con sus 
        productos específicos y así analizar si todas las convocatorias adjudicaron su energía o si hubo productos desiertos.
        """)

        Xdata = pd.read_excel('Datos/df_imputado.xlsx')

        with st.expander("📂 Ver Base de Datos Cruzada"):
            st.dataframe(Xdata.head(10))

        st.download_button("⬇️ Descargar CSV", Xdata.to_csv(index=False), "df_imputado.csv", "text/csv")

    st.divider()

    ## 🏆 SECCIÓN: Base de Datos Final (Preprocesada)
    with st.container():
        st.header("🏆 Base de Datos Final (Cruzada y Preprocesada)")
        st.markdown("""
        <div style= padding: 10px; border-radius: 10px;'>
            <ul>
                <li>✅ Se filtran solo las convocatorias adjudicadas.</li>
                <li>✅ Se eliminan columnas post-adjudicación para asegurarse de que el modelo aprenda solo a partir de información que está disponible antes de la adjudicación.</li>
                <li>✅ Se codifican variables categóricas.</li>
                <li>✅ Se normalizan los datos con MinMaxScaler.</li>
                <li>✅ Se dividen en conjuntos de entrenamiento, validación y prueba.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        df_final = pd.read_excel('Datos/df_final.xlsx')

        with st.expander("📂 Ver Base de Datos Final"):
            st.dataframe(df_final.head(10))

        st.download_button("⬇️ Descargar CSV", df_final.to_csv(index=False), "df_final.csv", "text/csv")
