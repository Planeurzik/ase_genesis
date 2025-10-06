import umap
import matplotlib.pyplot as plt
import numpy as np

def return_embedding_and_data(data_file, latent_dim):
    data = np.load(data_file)

    n_neighbors = 20

    if latent_dim == 3:
        n_neighbors = 30

    coords = data[:, :latent_dim]
    values = data[:, latent_dim:]

    keys, inv = np.unique(coords, axis=0, return_inverse=True)

    new_data = np.zeros((len(keys), latent_dim + 2))
    new_data[:, :latent_dim] = keys 
    for i in range(2): 
        new_data[:, latent_dim+i] = np.bincount(inv, weights=values[:, i]) / np.bincount(inv)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(new_data[:,:-2])
    return embedding, new_data

embedding1, new_data1 = return_embedding_and_data("lafan_16.npy", 16)
embedding2, new_data2 = return_embedding_and_data("data_resampling_z_6d.npy",6)
embedding3, new_data3 = return_embedding_and_data("data_3d_big.npy", 3)

new_data1[:, -1] = np.clip(new_data1[:, -1], -0.5, 0.5)
new_data2[:, -1] = np.clip(new_data2[:, -1], -0.5, 0.5)
new_data3[:, -1] = np.clip(new_data3[:, -1], -0.5, 0.5)

fig, axes = plt.subplots(2, 3, figsize=(18,10))

# First row: speed coloring
linear_colorbar = axes[0,0].scatter(
    embedding1[:, 0], embedding1[:, 1],
    c=new_data1[:, -2],
    cmap="coolwarm", s=10
)
cbar1 = fig.colorbar(linear_colorbar, ax=axes[0,0])
cbar1.set_label("Linear velocity")
#axes[0].set_title("UMAP projection colored by Speed")

# Second row: angular coloring
angular_colorbar = axes[1,0].scatter(
    embedding1[:, 0], embedding1[:, 1],
    c=new_data1[:, -1],
    cmap="viridis", s=10
)
cbar2 = fig.colorbar(angular_colorbar, ax=axes[1,0])
cbar2.set_label("Angular velocity")
#axes[1].set_title("UMAP projection colored by Angular Feature")

# First row: speed coloring
linear_colorbar = axes[0,1].scatter(
    embedding2[:, 0], embedding2[:, 1],
    c=new_data2[:, -2],
    cmap="coolwarm", s=10
)
cbar1 = fig.colorbar(linear_colorbar, ax=axes[0,1])
cbar1.set_label("Linear velocity")
#axes[0].set_title("UMAP projection colored by Speed")
# Second row: angular coloring
angular_colorbar = axes[1,1].scatter(
    embedding2[:, 0], embedding2[:, 1],
    c=new_data2[:, -1],
    cmap="viridis", s=10
)
cbar2 = fig.colorbar(angular_colorbar, ax=axes[1,1])
cbar2.set_label("Angular velocity")
#axes[1].set_title("UMAP projection colored by Angular Feature")

# First row: speed coloring
linear_colorbar = axes[0,2].scatter(
    embedding3[:, 0], embedding3[:, 1],
    c=new_data3[:, -2],
    cmap="coolwarm", s=10
)
cbar1 = fig.colorbar(linear_colorbar, ax=axes[0,2])
cbar1.set_label("Linear velocity")
#axes[0].set_title("UMAP projection colored by Speed")

# Second row: angular coloring
angular_colorbar = axes[1,2].scatter(
    embedding3[:, 0], embedding3[:, 1],
    c=new_data3[:, -1],
    cmap="viridis", s=10
)
cbar2 = fig.colorbar(angular_colorbar, ax=axes[1,2])
cbar2.set_label("Angular velocity")
#axes[1].set_title("UMAP projection colored by Angular Feature")

axes[0,0].set_title("Latent dim = 16")
axes[0,1].set_title("Latent dim = 6")
axes[0,2].set_title("Latent dim = 3")

plt.tight_layout()
plt.savefig("umap_all.png", dpi=300)
plt.show()
#exit(0)

embedding, new_data = return_embedding_and_data("data_rohan_6d.npy", 6)

new_data[:, -1] = np.clip(new_data[:, -1], -0.5, 0.5)

fig, axes = plt.subplots(2, 2, figsize=(12,10), sharex=False, sharey=False)

# First row: speed coloring
linear_colorbar = axes[0,0].scatter(
    embedding[:, 0], embedding[:, 1],
    c=new_data[:, -2],
    cmap="coolwarm", s=10
)
cbar1 = fig.colorbar(linear_colorbar, ax=axes[0,0])
cbar1.set_label("Linear velocity")

# Second row: angular coloring
angular_colorbar = axes[1,0].scatter(
    embedding[:, 0], embedding[:, 1],
    c=new_data[:, -1],
    cmap="viridis", s=10
)
cbar2 = fig.colorbar(angular_colorbar, ax=axes[1,0])
cbar2.set_label("Angular velocity")

linear_colorbar = axes[0,1].scatter(
    embedding2[:, 0], embedding2[:, 1],
    c=new_data2[:, -2],
    cmap="coolwarm", s=10
)
cbar1 = fig.colorbar(linear_colorbar, ax=axes[0,1])
cbar1.set_label("Linear velocity")
#axes[0].set_title("UMAP projection colored by Speed")
# Second row: angular coloring
angular_colorbar = axes[1,1].scatter(
    embedding2[:, 0], embedding2[:, 1],
    c=new_data2[:, -1],
    cmap="viridis", s=10
)
cbar2 = fig.colorbar(angular_colorbar, ax=axes[1,1])
cbar2.set_label("Angular velocity")

axes[0,0].set_title("Mocap dataset")
axes[0,1].set_title("Lafan dataset")

plt.tight_layout()
plt.savefig("umap_rohan_lafan_6d.png", dpi=300)
plt.show()

embedding4, new_data4 = return_embedding_and_data("real_data_vel_angle.npy", 6)

new_data4[:, -1] = np.clip(new_data4[:, -1], -0.5, 0.5)

fig, axes = plt.subplots(2, 2, figsize=(12,10), sharex=False, sharey=False)

linear_colorbar = axes[0,0].scatter(
    embedding4[:, 0], embedding4[:, 1],
    c=new_data4[:, -2],
    cmap="coolwarm", s=10
)
cbar1 = fig.colorbar(linear_colorbar, ax=axes[0,0])
cbar1.set_label("Linear velocity")

# Second row: angular coloring
angular_colorbar = axes[1,0].scatter(
    embedding4[:, 0], embedding4[:, 1],
    c=new_data4[:, -1],
    cmap="viridis", s=10
)
cbar2 = fig.colorbar(angular_colorbar, ax=axes[1,0])
cbar2.set_label("Angular velocity")

linear_colorbar = axes[0,1].scatter(
    embedding[:, 0], embedding[:, 1],
    c=new_data[:, -2],
    cmap="coolwarm", s=10
)
cbar1 = fig.colorbar(linear_colorbar, ax=axes[0,1])
cbar1.set_label("Linear velocity")
#axes[0].set_title("UMAP projection colored by Speed")
# Second row: angular coloring
angular_colorbar = axes[1,1].scatter(
    embedding[:, 0], embedding[:, 1],
    c=new_data[:, -1],
    cmap="viridis", s=10
)
cbar2 = fig.colorbar(angular_colorbar, ax=axes[1,1])
cbar2.set_label("Angular velocity")

axes[0,0].set_title("Training for deployment on real robot")
axes[0,1].set_title("Training in simulation")

plt.tight_layout()
#plt.savefig("umap_real_rohan_6d.png", dpi=300)
plt.show()