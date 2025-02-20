import streamlit as st
from secciones import inicio, lineales, no_lineales, tabnet, propuesta  # Importar las secciones de la carpeta "pages"
from sklearn.metrics import r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np

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

st.sidebar.title("Índice")
sections = ["Inicio", "Regresores Clásicos Lineales", "Regresores Clásicos No Lineales", "TabNet", "Propuesta"]
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