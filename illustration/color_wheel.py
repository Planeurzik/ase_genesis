import numpy as np
import matplotlib.pyplot as plt
import colorsys

# Taille de l'image
res = 500
radius = 1.0

# Grille de coordonnées
x = np.linspace(-radius, radius, res)
y = np.linspace(-radius, radius, res)
X, Y = np.meshgrid(x, y)

# Calcul du masque du cercle
mask = X**2 + Y**2 <= radius**2

# Initialisation de l'image RGB (forme: H, W, 3)
image = np.ones((res, res, 3))

# Calcul de l'angle pour chaque pixel
theta = np.arctan2(Y, X)  # angle in radians from -π to π
hue = (theta + np.pi) / (2 * np.pi)  # normalized to [0, 1]

# Applique HSV -> RGB uniquement sur les pixels dans le cercle
for i in range(res):
    for j in range(res):
        if mask[i, j]:
            r, g, b = colorsys.hsv_to_rgb(hue[i, j], 1.0, 1.0)
            image[i, j] = [r, g, b]
        else:
            image[i, j] = [1, 1, 1]  # blanc hors du cercle

# Exemple de courbe sinusoidale
angles = np.linspace(0, 2*np.pi, 1000)
values = np.abs(np.sin(angles))   # scale sinusoid so it fits inside circle
radii = radius + values         # radius = base circle + sinusoid

# Conversion en coordonnées cartésiennes
x_curve = radii * np.cos(angles)
y_curve = radii * np.sin(angles)

# Affichage
plt.figure(figsize=(6, 6))
plt.imshow(image[::-1], extent=[-1, 1, -1, 1])
plt.plot(x_curve, y_curve, color="black", linewidth=2)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.grid(False)
plt.show()