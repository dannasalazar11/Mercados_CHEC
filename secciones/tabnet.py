import streamlit as st
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np
import joblib

def my_r2_score_fn(y_pred, y_true):
    total_variance = torch.var(y_true, unbiased=False)
    unexplained_variance = torch.mean((y_true - y_pred) ** 2)
    r2_score = unexplained_variance / total_variance
    return r2_score

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

@st.cache_resource
def load_model():
    model = CustomTabNetRegressor()  # Instancia el modelo vacío
    model.network.load_state_dict(torch.load("Modelos/custom_tabnet_model.pth", map_location=torch.device('cpu')))  # Cargar pesos
    model.eval()  # Poner en modo evaluación
    return model

def mostrar():
    st.title("TabNet")

    # Mostrar una imagen desde una ruta local
    st.image("Datos/Imagenes/tabnet.jpg", caption="Arquitectura de la TabNet", use_container_width=True)

    clf = load_model()
    st.write("Modelo cargado")


  