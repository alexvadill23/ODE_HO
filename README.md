# ODE_HO  
Solving the harmonic oscillator equation using PINNs: comparison with standard neural networks and exploration of inverse problems.

This repository contains the code developed to solve a second-order ordinary differential equation (ODE) using **Physics-Informed Neural Networks (PINNs)**. It includes comparisons with standard neural networks using different datasets, as well as an example of an inverse problem.

We consider the following ODE:

\[
\frac{d^2y}{dx^2} + y = 0, \quad y(0) = 1, \quad y'(0) = 0
\]

The exact (analytical) solution is:

\[
y(x) = \cos(x)
\]

To explore the capability of Physics-Informed Neural Networks compared to ordinary Neural Networks, we analyze three different cases:

- **Case 1:** abundant training points in the domain \([0, 2\pi]\). We evaluate extrapolation ability outside the training domain.
- **Case 2:** minimal training points in the domain \([0, 2\pi]\), testing accuracy with limited data.
- **Case 3:** training using only boundary conditions (BC).

For all cases, the following architecture and hyperparameters are used:

| Parameter             | Value         |
|-----------------------|---------------|
| Number of layers      | 5             |
| Neurons per layer     | 32            |
| Activation function   | Tanh          |
| Learning rate         | 1e-3          |
| Optimizer             | Adam          |
| Number of epochs      | 3000          |
| PDE points            | 100           |

The loss function is composed of data and PDE parts. The PINN optimizes both, while the standard NN optimizes only the data part:

\[
L = \lambda_{\text{DATA}} L_{\text{DATA}} + \lambda_{\text{PDE}} L_{\text{PDE}}
\]

where \(L_{\text{DATA}}\) measures the error on experimental data, and \(L_{\text{PDE}}\) is the residual of the differential equation evaluated at collocation points. The weights are fixed as \(\lambda_{\text{DATA}} = \lambda_{\text{PDE}} = 0.5\), giving equal importance to both terms.

---

## 📊 Results

### Case 1: Abundant training data in \([0, 2\pi]\)
![PINN vs NN vs cos(x)](comparacion_funciones_completa.png)

| Interval              | PINN MSE    | Simple NN MSE  |
|-----------------------|-------------|----------------|
| Training \([0, 2\pi]\)         | 0           | 0.00007        |
| Extrapolation \([2\pi, 4\pi]\) | 0.00002     | 1.92391        |

---

### Case 2: Minimal training data in \([0, 2\pi]\)

| Interval              | PINN MSE    | Simple NN MSE  |
|-----------------------|-------------|----------------|
| Training \([0, 2\pi]\)         | 0           | 1.99689        |
| Extrapolation \([2\pi, 4\pi]\) | 0           | 5.13736        |

---

### Case 3: Only boundary conditions at \(x = 0\)

| Interval              | PINN MSE    | Simple NN MSE  |
|-----------------------|-------------|----------------|
| Training \([0, 2\pi]\)         | 0.00048     | 1.40306        |
| Extrapolation \([2\pi, 4\pi]\) | 0.00161     | 1.43553        |







****
