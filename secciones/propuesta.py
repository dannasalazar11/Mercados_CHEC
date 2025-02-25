import streamlit as st

def mostrar():
    # Título del Proyecto
    st.title("\U0001F4C8 Propuesta: Predicción Inteligente de Precios por Producto en Convocatorias de Compra de Energía")

    # Introducción
    st.markdown(
        """
        Este proyecto se enfoca en estimar de forma precisa el precio de **cada producto específico** incluido en las convocatorias de compra de energía de la CHEC, utilizando **TabNet** para el análisis de datos tabulares y **análisis de grafos** para capturar las interacciones entre múltiples variables y productos.
        """
    )

    st.markdown("---")

    # Datos y Entrenamiento
    st.header("Datos y Entrenamiento")
    st.markdown(
        """
        - **Datos Históricos de SICEP**: Se entrenará el modelo utilizando la información histórica de convocatorias, precios, características de los productos y adjudicaciones anteriores.
        - **Respuestas de “Adendas”**: Incluir la información de las aclaraciones y modificaciones realizadas en las convocatorias, ya que pueden afectar los precios finales.
        - **Variables Económicas (posibles ejemplos)**
        - **Tasa de cambio (USD/COP)**: Impacto de la fluctuación del dólar sobre costos de importación.
        - **Salario Mínimo Vigente (SMV)**: Costos de mano de obra o contratación.
        - **Índice de Precios al Consumidor (IPC)**: Evolución de costos de producción y distribución.
        """
    )

    st.markdown(
        "Incluir estas variables económicas ayuda a **capturar de manera más realista** la dinámica del mercado energético, considerando factores macroeconómicos y laborales."
    )

    st.markdown("---")

    # Objetivos Específicos
    st.header("Objetivos Específicos")
    st.markdown(
        """
        1. **Recopilar y limpiar** la información histórica de convocatorias y Adendas, garantizando la calidad de los datos.
        2. **Entrenar TabNet** para predecir precios específicos por producto, identificando las variables más influyentes.
        3. **Analizar la relación entre productos y variables** mediante un modelo de grafos.
        4. **Ajustar hiperparámetros** para maximizar la precisión del modelo.
        5. **Implementar un dashboard** donde la CHEC pueda ingresar nuevas convocatorias y obtener predicciones detalladas.
        """
    )

    st.markdown("---")

    # Resultados Esperados
    st.header("Resultados Esperados")
    st.markdown(
        """
        - **Optimización de precios por producto**, incrementando la adjudicación exitosa.
        - **Mayor explicabilidad** gracias a TabNet, mostrando la influencia de cada variable.
        - **Toma de decisiones estratégicas** considerando escenarios económicos.
        - **Visión integral** de cómo factores externos y detalles de convocatorias afectan la negociación de precios.
        """
    )

    st.markdown(
        "Con esta propuesta, la CHEC podrá **maximizar la eficiencia en sus procesos de compra de energía**, integrando factores económicos y Adendas para una **mejor toma de decisiones**."
    )

    st.image("Datos/Imagenes/Mercados CHEC.png", caption="", use_container_width=True)
