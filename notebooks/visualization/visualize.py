import cv2
import numpy as np
import pyvista as pv
import os

if 1:
    data_dir = '/home/mithil/PycharmProjects/SenNetKideny/data'
    file = [f'{data_dir}/train/kidney_1_dense/labels/{i:04d}.tif' for i in
            range(len(os.listdir(f'{data_dir}/train/kidney_1_dense/labels/')[1000:2000]))]
    mask = []
    for i, f in enumerate(file):
        print('\r', i, end='')
        v = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        mask.append(v)
    mask = np.stack(mask)
    print('')
    mask = mask / 255
    print(mask.shape)
    # np.save('truth.npy',mask)

pl = pv.Plotter(notebook=False)
point1 = np.stack(np.where(mask > 0.1)).T
pd1 = pv.PolyData(point1)
mesh1 = pd1.glyph(geom=pv.Cube())
pl.add_mesh(mesh1, color='blue')

pl.show()
