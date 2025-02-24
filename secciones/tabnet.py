import streamlit as st

def mostrar():
    st.title("TabNet")

    # Mostrar una imagen desde una ruta local
    st.image("Datos/Imagenes/tabnet.jpg", caption="Arquitectura de la TabNet", use_container_width=True)

    st.markdown("""TabNet es un modelo de aprendizaje profundo diseñado específicamente para el 
    procesamiento de datos tabulares, combinando la eficiencia de los modelos basados en árboles 
    con la capacidad de representación de las redes neuronales profundas. Su arquitectura se basa 
    en un mecanismo de atención secuencial que selecciona de manera adaptativa las características 
    más relevantes en cada paso del proceso de aprendizaje, optimizando así la interpretación y el
    rendimiento del modelo. \n Una de sus principales ventajas es su capacidad de interpretabilidad, ya que permite 
    visualizar qué variables han tenido mayor impacto en las predicciones, facilitando la 
    comprensión del proceso de toma de decisiones. Además, TabNet es capaz de aprender 
    directamente de los datos sin requerir una preprocesamiento extenso, lo que lo hace una opción 
    eficiente y flexible en diversos ámbitos, como el sector financiero, la salud y la analítica 
    empresarial.""")

    st.markdown('<a href="https://colab.research.google.com/drive/1dDS0gcXYdsSh4iuUfptsn_S-uBmE7YNi?usp=sharing" target="_blank"><button style="padding:10px 20px;font-size:16px;">Ir al cuaderno de entrenamiento</button></a>', unsafe_allow_html=True)



  