import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load("velocity_angle_3.npy")

coords = data[:, :3]
values = data[:, 3:]

keys, inv = np.unique(coords, axis=0, return_inverse=True)

new_data = np.zeros((len(keys), 5))
new_data[:, :3] = keys 
for i in range(2): 
    new_data[:, 3+i] = np.bincount(inv, weights=values[:, i]) / np.bincount(inv)

# Cartesian coords of dataset points
x = new_data[:,0]
y = new_data[:,1]
z = new_data[:,2]

# Velocities
lin_vel = new_data[:,3]
ang_vel = new_data[:,4]
#ang_vel = np.clip(ang_vel, -0.4, 0.4)

# ----------------------------
# Visualization
# ----------------------------
fig = plt.figure(figsize=(10, 10))

# Subplot (1,1): linear velocity
ax = fig.add_subplot(223, projection='3d')
p = ax.scatter(x, y, z, c=lin_vel, cmap='coolwarm', s=30)
fig.colorbar(p, ax=ax, shrink=0.6, label="Linear Velocity")
ax.set_title("Dataset linear velocity")

# Subplot (1,2): angular velocity
ax = fig.add_subplot(224, projection='3d')
p = ax.scatter(x, y, z, c=ang_vel, cmap='viridis', s=30)
fig.colorbar(p, ax=ax, shrink=0.6, label="Angular Velocity")
ax.set_title("Dataset angular velocity")

# Generate perfect sphere grid
n_theta, n_phi = 20, 40
theta = np.linspace(0, np.pi, n_theta)   # polar angle
phi = np.linspace(0, 2*np.pi, n_phi)     # azimuth
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

x_sphere = np.sin(theta_grid) * np.cos(phi_grid)
y_sphere = np.sin(theta_grid) * np.sin(phi_grid)
z_sphere = np.cos(theta_grid)

c = np.cos(phi_grid) + 1

# Subplot (2,1): placeholder for toy dataset
ax = fig.add_subplot(221, projection='3d')
p = ax.scatter(x_sphere, y_sphere, z_sphere, c=c, cmap='coolwarm', s=30)
fig.colorbar(p, ax=ax, shrink=0.6, label="Linear Velocity")
ax.set_title("Theoretical linear velocity")

c = np.cos(theta_grid)

# Subplot (2,2): perfect sphere grid
ax = fig.add_subplot(222, projection='3d')
p = ax.scatter(x_sphere, y_sphere, z_sphere, c=c, cmap='viridis', s=30)
fig.colorbar(p, ax=ax, shrink=0.6, label="Angular Velocity")
ax.set_title("Theoretical angular velocity")

plt.tight_layout()
plt.savefig("3d_latent_spaces.png", dpi=300)
plt.show()
