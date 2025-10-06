import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import vonmises

# Function to generate points on the surface of a sphere using von Mises-Fisher distribution
def generate_vmf_points(mean_direction, kappa, num_points):
    # Generate points using von Mises distribution for azimuthal angle
    phi = vonmises.rvs(kappa, size=num_points) + mean_direction[0]
    # Generate points using normal distribution for polar angle
    theta = np.random.normal(mean_direction[1], 1/np.sqrt(kappa), num_points)
    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack((x, y, z))

# Mean directions for clusters (azimuthal and polar angles in radians)
mean_directions = [
    (np.pi/4, -np.pi/4),  # Cluster 1
    (np.pi/2, -np.pi/2),  # Cluster 2
    (3*np.pi/4, -np.pi/3) # Cluster 3
]

# Concentration parameter (kappa)
kappa = 50

# Number of points per cluster
num_points_per_cluster = 50

# Generate points for each cluster
points_cluster_1 = generate_vmf_points(mean_directions[0], kappa, num_points_per_cluster)
points_cluster_2 = generate_vmf_points(mean_directions[1], kappa, num_points_per_cluster)
points_cluster_3 = generate_vmf_points(mean_directions[2], kappa, num_points_per_cluster)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster with different colors
ax.scatter(points_cluster_1[0], points_cluster_1[1], points_cluster_1[2], color='red', label='Walk')
ax.scatter(points_cluster_2[0], points_cluster_2[1], points_cluster_2[2], color='blue', label='Stand up')
ax.scatter(points_cluster_3[0], points_cluster_3[1], points_cluster_3[2], color='green', label='Push forward with the arms')

# Plot the sphere
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="grey", alpha=0.1)

# Adding labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.savefig("sphere_clusters.png", dpi=300)
plt.show()
