import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# Create toy dataset
# ----------------------------
n_theta, n_phi = 20, 40  # resolution
theta = np.linspace(0, np.pi, n_theta)   # polar angle (0 = north pole)
phi = np.linspace(0, 2*np.pi, n_phi)     # azimuth

theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

# Cartesian coords of sphere points
x = np.sin(theta_grid) * np.cos(phi_grid)
y = np.sin(theta_grid) * np.sin(phi_grid)
z = np.cos(theta_grid)

# Define toy velocities (last 2 dims)
# For simplicity: linear velocity = cos(theta), angular velocity = sin(phi)
lin_vel = np.cos(theta_grid)
ang_vel = np.sin(phi_grid)

# Shape: (n_theta, n_phi, 4)
data = np.stack([theta_grid, phi_grid, lin_vel, ang_vel], axis=-1)

# ----------------------------
# Visualization 1: Sphere with color
# ----------------------------
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
p = ax.scatter(x, y, z, c=lin_vel, cmap='coolwarm', s=30)
fig.colorbar(p, ax=ax, shrink=0.6, label="Linear Velocity")
ax.set_title("Sphere colored by linear velocity")

ax = fig.add_subplot(122, projection='3d')
p = ax.scatter(x, y, z, c=ang_vel, cmap='viridis', s=30)
fig.colorbar(p, ax=ax, shrink=0.6, label="Angular Velocity")
ax.set_title("Sphere colored by angular velocity")

plt.tight_layout()

plt.show()
""" 
# ----------------------------
# Visualization 2: Projection with arrows
# ----------------------------
plt.figure(figsize=(6,6))
plt.quiver(phi_grid, theta_grid, lin_vel, ang_vel, angles='xy', scale=20)
plt.xlabel("Azimuth (phi)")
plt.ylabel("Polar (theta)")
plt.title("Velocity vectors in angle space")
plt.show() """
