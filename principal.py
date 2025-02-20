import streamlit as st
import torch
from secciones import inicio, lineales, no_lineales, tabnet, propuesta  # Importar las secciones
from sklearn.metrics import r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np

# 📌 Función personalizada de R²
def my_r2_score_fn(y_pred, y_true):
    total_variance = torch.var(y_true, unbiased=False)
    unexplained_variance = torch.mean((y_true - y_pred) ** 2)
    return 1 - (unexplained_variance / total_variance)

# 📌 Clase personalizada de TabNet
class CustomTabNetRegressor(TabNetRegressor):
    def __init__(self, *args, **kwargs):
        super(CustomTabNetRegressor, self).__init__(*args, **kwargs)

    def forward(self, X):
        output, M_loss = self.network(X)
        output = torch.relu(output)
        return output, M_loss

    def predict(self, X):
        device = next(self.network.parameters()).device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(device)
        with torch.no_grad():
            output, _ = self.forward(X)
        return output.cpu().numpy()

# 📌 Configuración de la página
st.set_page_config(page_title="Modelos de Predicción", layout="wide")

# 📌 Sidebar con navegación
st.sidebar.title("Índice de Secciones")
sections = {
    "Inicio": "Inicio",
    "Regresores Clásicos Lineales": "Regresores Lineales",
    "Regresores Clásicos No Lineales": "Regresores No Lineales",
    "TabNet": "TabNet",
    "Propuesta": "Propuesta"
}

choice = st.sidebar.radio("Selecciona una sección", list(sections.keys()), format_func=lambda x: sections[x])

# 📌 Mostrar la sección seleccionada
if choice == "Inicio":
    st.success("Bienvenido. Selecciona una sección en el menú lateral para comenzar.")
    inicio.mostrar()

elif choice == "Regresores Clásicos Lineales":
    st.info("Sección de regresores clásicos lineales.")
    lineales.mostrar()

elif choice == "Regresores Clásicos No Lineales":
    st.info("Sección de regresores clásicos no lineales.")
    no_lineales.mostrar()

elif choice == "TabNet":
    st.info("Sección de TabNet para modelos de predicción avanzados.")
    tabnet.mostrar()

elif choice == "Propuesta":
    st.info("Sección con la propuesta de modelo.")
    propuesta.mostrar()
