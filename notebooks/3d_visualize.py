import cv2
import numpy as np
import pyvista as pv
import os
from tqdm import tqdm
# Load data
data_dir = '/home/mithil/PycharmProjects/SenNetKideny/data'
files = [f'{data_dir}/train/kidney_1_dense/labels/{i:04d}.tif' for i in
         range(1000, 2000)]
mask = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
mask = np.stack(mask) / 255

# Create 3D visualization
point1 = np.stack(np.where(mask > 0.1)).T
centroid = np.mean(point1, axis=0)

pd1 = pv.PolyData(point1)
mesh1 = pd1.glyph(geom=pv.Cube())

# Set up the plotter and open a movie file
filename = "kidney_visualization.mp4"
plotter = pv.Plotter()
plotter.open_movie(filename)

# Add the mesh
plotter.add_mesh(mesh1, color='blue')

# Start the plotter

# Write the initial frame
plotter.write_frame()

# Animation parameters
frame_count = 360
distances = np.linalg.norm(point1 - centroid, axis=1)
radius = np.max(distances) * 1.5  # 1.5 is a scaling factor for better visibility


# Update the camera position and write each frame
for i in tqdm(range(frame_count)):
    angle = i * (360 / frame_count)
    x = radius * np.cos(np.radians(angle))
    y = radius * np.sin(np.radians(angle))
    camera_position = [x, y, 120]  # Adjust the Z value as needed
    focal_point = centroid.tolist()  # Convert centroid to a list if it's a numpy array
    view_up = [0, 0, 1]  # Z-axis is up

    plotter.camera_position = (camera_position, focal_point, view_up)
    plotter.add_text(f"Iteration: {i}", name='time-label')

    plotter.write_frame()

# Close the plotter to finish the movie
plotter.close()
