import matplotlib.pyplot as plt
import os
import numpy as np

#source ~/miniconda3/etc/profile.d/conda.sh
path = "../datasets/HO3D/train/ABF10"

rgb_path = os.path.join(path, "rgb")
rgb = plt.imread(rgb_path+"/0010.png")

seg_path = os.path.join(path, "seg")
seg = plt.imread(seg_path+"/0010.jpg")
seg = seg >=150
seg = seg.astype(np.uint8)*255
print(seg.shape)
print(np.unique(seg[:, :, 0]))
print(np.unique(seg[:, :, 1]))
print(np.unique(seg[:, :, 2]))
plt.imsave("save.png", seg)
# print(list(seg[:, :, 1]))