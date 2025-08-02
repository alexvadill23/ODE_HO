import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os

# Configuración de reproducibilidad COMPLETA
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = '42'

def set_pi_ticks(ax, xmax):
    ticks = np.arange(0, xmax + np.pi, np.pi)
    labels = [ "0" if i==0 else rf"${i}\pi$" for i in range(len(ticks)) ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))
        out = self.activation(self.fc5(out))
        out = self.fc6(out)
        return out

    def compute_derivatives(self, x):
        x = x.clone().detach().requires_grad_(True)
        y = self(x)
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True)[0]
        return y, dy_dx, d2y_dx2

# Configuración del experimento
epochs = 3000
N_data = 40
N_pde = 100

x_pde_orig = np.linspace(0, 4*np.pi, N_pde).reshape(-1, 1)
        

x_total = np.random.uniform(0, 4*np.pi, (N_data, 1))
y_total = np.cos(np.sqrt(2)*x_total)
        
x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=42)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
        
x_pde_t = torch.tensor(x_pde_orig, dtype=torch.float32)

# Ponderaciones y parámetro
w_pde = 0.5
w_data = 0.5
a_param = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

# Modelo y optimizador (configuración idéntica)
model_PINN = PINN()
optimizer_PINN = optim.Adam(list(model_PINN.parameters()) + [a_param], lr=0.001)
        
loss_train_list = []
loss_test_list = []
loss_pde_list = []
loss_data_list = []
a_evolution = []     

print("=== INICIANDO ENTRENAMIENTO PARA DESCUBRIMIENTO DE PARÁMETROS ===")
print(f"Parámetro real: a = 2.0")
print(f"Parámetro inicial: a = {a_param.item():.6f}")
print(f"Datos de entrenamiento: {len(x_train)} puntos en [0,4pi]")  
print(f"Puntos PDE: {N_pde} puntos en [0,4pi]\n")

for epoch in range(epochs):
    model_PINN.train()
    optimizer_PINN.zero_grad()
            
    y_pde, dy_dx, d2y_dx2 = model_PINN.compute_derivatives(x_pde_t)
    loss_pde = torch.mean((d2y_dx2 + a_param * y_pde)**2)
            
    y_pred_data = model_PINN(x_train)
    loss_data = torch.mean((y_pred_data - y_train)**2)

    loss = w_pde * loss_pde + w_data * loss_data
            
    loss.backward()
    optimizer_PINN.step()

    a_evolution.append(a_param.item())
    loss_pde_list.append(loss_pde.item())
    loss_data_list.append(loss_data.item())
    loss_train_list.append(loss.item())
            
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}")
        print(f"           Loss PDE = {loss_pde.item():.6f}, Loss Data = {loss_data.item():.6f}")
        print(f"           a_param = {a_param.item():.6f}\n")

# Evaluación final
model_PINN.eval()
with torch.no_grad():
    # Predicciones en el conjunto de test
    y_pred_test = model_PINN(x_test)
    mse_test = torch.mean((y_pred_test - y_test)**2).item()
    
    # Predicciones para la visualización
    x_fine = np.linspace(0, 4*np.pi, 1000).reshape(-1, 1)
    x_fine_t = torch.tensor(x_fine, dtype=torch.float32)
    y_PINN = model_PINN(x_fine_t).detach().numpy()
    
    # Solución analítica real (con a=2)
    y_real = np.cos(np.sqrt(2) * x_fine)
    
    # MSE global (ya no distinguimos entre intervalos)
    x_eval = np.linspace(0, 4*np.pi, 1000).reshape(-1, 1)
    x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
    y_eval_PINN = model_PINN(x_eval_t).detach().numpy()
    y_eval_real = np.cos(np.sqrt(2) * x_eval)
    
    mse_global = mean_squared_error(y_eval_real, y_eval_PINN)

# Resultados finales
print(f"\n=== RESULTADOS FINALES ===")
print(f"Parámetro real: a = 2.000")
print(f"Parámetro estimado: a = {a_param.item():.6f}")
print(f"Error relativo: {abs(a_param.item() - 2.0)/2.0 * 100:.3f}%")
print(f"Error absoluto: {abs(a_param.item() - 2.0):.6f}")
print(f"\n--- MÉTRICAS DE PRECISIÓN ---")
print(f"MSE Test: {mse_test:.8f}")
print(f"MSE Global [0,4pi]: {mse_global:.8f}")

# GRÁFICAS SEPARADAS

# 1. Evolución del parámetro a
plt.figure(figsize=(10, 6))
plt.plot(a_evolution, 'b-', linewidth=2, label=f'a estimado')
plt.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='a real = 2.0')
plt.xlabel("Época")
plt.ylabel("Valor de a")
plt.title("Evolución del parámetro a durante el entrenamiento")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('parameter_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Evolución de las pérdidas
plt.figure(figsize=(10, 6))
plt.plot(loss_train_list, 'b-', label='Loss Total', linewidth=1.5)
plt.plot(loss_pde_list, 'r-', label='Loss PDE', linewidth=1.5)
plt.plot(loss_data_list, 'g-', label='Loss Data', linewidth=1.5)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Evolución de las funciones de pérdida")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig('parameter_losses.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Comparación de soluciones
plt.figure(figsize=(12, 6))
plt.plot(x_fine, y_PINN, 'b-', label=f'PINN (a={a_param.item():.4f})', linewidth=2)
plt.plot(x_fine, y_real, 'k--', label='Solución real: cos(√2·x)', linewidth=2)

# Puntos de entrenamiento SOBRE la función (no en y=0)
x_train_np = x_train.detach().numpy()
y_train_np = y_train.detach().numpy()
plt.scatter(x_train_np, y_train_np, color='red', marker='s', 
           s=60, label='Puntos de entrenamiento', zorder=6, edgecolors='darkred')

# Puntos PDE en y=0
plt.scatter(x_pde_orig, np.zeros_like(x_pde_orig), color='green', marker='o', 
           s=25, label='Puntos PDE', zorder=5, alpha=0.6)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Descubrimiento de Parámetros: PINN vs Solución Analítica')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
set_pi_ticks(plt.gca(), 4*np.pi)
plt.savefig('parameter_solution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Guardar resultados
resultados = {
    'Parámetro real': 2.0,
    'Parámetro estimado': a_param.item(),
    'Error relativo (%)': abs(a_param.item() - 2.0)/2.0 * 100,
    'Error absoluto': abs(a_param.item() - 2.0),
    'MSE Test': mse_test,
    'MSE Global': mse_global,
    'Épocas totales': epochs
}

