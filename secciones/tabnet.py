import streamlit as st
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
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

# Cargar el modelo
def load_model():
    model_path = "Modelos/Best_model.pth"  # Ruta del modelo guardado
    model = torch.load(model_path, weights_only=False)  # Cargar el modelo
    model.eval()  # Poner en modo evaluación
    return model

def R2(clf):
    # Start figure for 1 row and 3 columns
    y_pred=clf.predict(X_test)
    # y_pred=np.tile(y_pred, (1, 2))
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1
    r2_1 = r2_score(y_test, y_pred)
    axs[0].scatter(y_test, y_pred, alpha=0.5)
    axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', label=f'R² = {r2_1:.2f}')
    axs[0].set_title('Valores esacalados: etiqueta verdadera vs estimada')
    axs[0].set_xlabel('Etiqueta verdadera')
    axs[0].set_ylabel('Etiqueta estimada')
    axs[0].legend()

    # Concatenar X_test con y_pred para hacer la transformación inversa
    data_pred = np.concatenate((X_test, y_pred.reshape(-1, 1)), axis=1)
    data_test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)

    # Aplicar inverse_transform a toda la matriz
    data_pred_inv = scaler.inverse_transform(data_pred)
    data_test_inv = scaler.inverse_transform(data_test)

    # Extraer solo las columnas de y después de la transformación inversa
    y_pred = data_pred_inv[:, -1]  # Última columna
    y_test_inv = data_test_inv[:, -1]  # Última columna

    # Plot 2
    r2_2 = r2_score(y_test_inv, y_pred)
    axs[1].scatter(y_test_inv, y_pred, alpha=0.5)
    axs[1].plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r-', label=f'R² = {r2_2:.2f}')
    axs[1].set_title('Valores sin esacalar: etiqueta verdadera vs estimada')
    axs[1].set_xlabel('Etiqueta verdadera (COP/kWh)')
    axs[1].set_ylabel('Etiqueta estimada (COP/kWh)')
    axs[1].legend()
    st.pyplot(fig)


def mostrar():
    st.title("TabNet")

    # Mostrar una imagen desde una ruta local
    st.image("Datos/Imagenes/tabnet.jpg", caption="Arquitectura de la TabNet", use_container_width=True)

    clf = load_model()


  