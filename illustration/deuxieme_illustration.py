import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
np.random.seed(0)
num_samples = 500
mean = np.array([[0.0, 0.0],[1.5, 1.2], [-0.8,0.2]])  # Gaussian mean
cov = np.array([[0.05, 0.0], [0.0, 0.05]])  # Covariance
radius = 1.0  # Hypersphere radius

nb_row = 1

# --- Create figure with 3 rows and 2 columns ---
fig, axes = plt.subplots(nb_row, 2, figsize=(10,10))

xlim = (-2.0, 2.0)
ylim = (-2.0, 2.0)

for row in range(1, nb_row+1):
    # --- Sample from Gaussian ---
    gaussian_samples = np.random.multivariate_normal(mean[row], cov, num_samples)

    # --- Project samples onto hypersphere ---
    norms = np.linalg.norm(gaussian_samples, axis=1, keepdims=True)
    sphere_samples = gaussian_samples / norms * radius

    # Left column: Gaussian samples + circle
    ax = axes[0]
    circle = plt.Circle((0, 0), radius, color='green', fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.scatter(gaussian_samples[:, 0], gaussian_samples[:, 1], alpha=0.3,
               label=r"$\overline{\mathbf{z}} \sim \mathcal{N}(\mu_\omega(\mathbf{s},g), \Sigma_\omega)$")
    ax.set_aspect('equal', 'box')
    #ax.set_title(f"Latent space with Gaussian #{row+1}")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, labelleft=False)

    # Right column: Projection onto hypersphere + circle
    ax = axes[1]
    circle = plt.Circle((0, 0), radius, color='green', fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.scatter(sphere_samples[:, 0], sphere_samples[:, 1], alpha=0.3,
               label=r"$\mathbf{z} = \frac{\overline{\mathbf{z}}}{||\overline{\mathbf{z}}||}$")
    ax.set_aspect('equal', 'box')
    #ax.set_title(f"Projection onto hypersphere #{row+1}")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, labelleft=False)


plt.tight_layout()
plt.savefig("second.png", dpi=300)
plt.show()
