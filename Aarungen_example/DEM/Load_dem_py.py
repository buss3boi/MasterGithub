import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Input file
filename = '6602_2_10m_z33.dem'

# Coordinates for the 'window' you want to 'slice out' from the dataset
wminx = 254100
wmaxx = 268000
wminy = 6620100
wmaxy = 6628700

# Grid spacing
dx = 10
dy = 10

wxvec = np.arange(wminx, wmaxx + dx, dx)
wyvec = np.arange(wminy, wmaxy + dy, dy)

wxgrid, wygrid = np.meshgrid(wxvec, wyvec)

# Read the raster data
with rasterio.open(filename) as src:
    Z = src.read(1)
    transform = src.transform

# Flip the Z matrix
Z = np.flipud(Z)

xv = np.arange(transform[0], transform[0] + transform[1] * Z.shape[1], dx)
yv = np.arange(transform[3], transform[3] + transform[5] * Z.shape[0], dy)
xcoord, ycoord = np.meshgrid(xv, yv)

minx, maxx, miny, maxy = np.min(xcoord), np.max(xcoord), np.min(ycoord), np.max(ycoord)

# Find indices of the crop area
i, j = np.where((xcoord >= wminx) & (xcoord <= wmaxx) & (ycoord >= wminy) & (ycoord <= wmaxy))

xc_ut = xcoord[i, j]
yc_ut = ycoord[i, j]
Zut = Z[i, j]

wi, wj = wxgrid.shape
wZ = Zut.reshape(wi, wj)

# Plot the original and cropped images
plt.figure()
plt.imshow(Z, extent=[minx, maxx, miny, maxy], cmap='viridis')
plt.gca().invert_yaxis()
plt.colorbar()
plt.plot(wminx, wminy, 'xr')
plt.plot(wmaxx, wmaxy, 'xr')
plt.axis('equal')

plt.figure()
plt.imshow(wZ, extent=[wminx, wmaxx, wminy, wmaxy], cmap='viridis')
plt.gca().invert_yaxis()
plt.colorbar()
plt.plot(wminx, wminy, 'xr')
plt.plot(wmaxx, wmaxy, 'xr')
plt.axis('equal')

plt.show()
