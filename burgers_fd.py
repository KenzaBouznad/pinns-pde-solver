import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import pandas as pd

# Parameters
L = 2.0  # Domain length from -1 to 1
Nx = 101  # Increased number of spatial points
Nt = 1001  # Increased number of time points
dx = L / (Nx - 1)  # Grid spacing
T = 1.0  # Final time
dt = T / (Nt - 1)  # Time step

# Physical parameters
nu = 0.1  # Viscosity

# Create spatial and temporal grids
x = np.linspace(-1, 1, Nx)
t = np.linspace(0, T, Nt)

# Initial Condition: u(x,0) = -sin(pi*x)
u = -np.sin(np.pi * x)
u_new = np.copy(u)

# Results storage
results = []

# Save initial condition for all x
for i in range(Nx):
    results.append([x[i], 0.0, u[i]])

# Time-stepping loop using Runge-Kutta 2nd Order
for n in range(1, Nt):
    # Predictor step (Euler)
    u_star = np.copy(u)
    for i in range(1, Nx - 1):
        conv = u[i] * (u[i + 1] - u[i - 1]) / (2 * dx)  # Convection term
        diff = nu * (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ** 2)  # Diffusion term
        u_star[i] = u[i] - dt * (conv - diff)

    # Corrector step
    for i in range(1, Nx - 1):
        conv = u_star[i] * (u_star[i + 1] - u_star[i - 1]) / (2 * dx)  # Convection term
        diff = nu * (u_star[i + 1] - 2 * u_star[i] + u_star[i - 1]) / (dx ** 2)  # Diffusion term
        u_new[i] = 0.5 * (u[i] + u_star[i] - dt * (conv - diff))

    # Apply boundary conditions
    u_new[0] = 0
    u_new[-1] = 0

    # Check for numerical instability
    if np.any(np.isnan(u_new)) or np.any(np.isinf(u_new)):
        print(f"Numerical instability detected at step {n}")
        break

    # Update the solution
    u[:] = u_new[:]

    # Save results for all x at current time
    for i in range(Nx):
        results.append([x[i], t[n], u[i]])

# Save results to CSV
with open("./results/results_fdm.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x", "t", "u"])
    writer.writerows(results)

# Load dataset for visualization
df_fd = pd.read_csv('./results/results_fdm.csv')

# Create heatmap
plt.figure(figsize=(10, 6))
pivot_table = df_fd.pivot(index='t', columns='x', values='u')
plt.imshow(pivot_table, extent=[df_fd['x'].min(), df_fd['x'].max(), 
                               df_fd['t'].min(), df_fd['t'].max()], 
           aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Heat Map of u at Different t and x (Finite Difference, RK2)')
plt.show()
