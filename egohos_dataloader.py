import matplotlib.pyplot as plt
import os
import numpy as np

#source ~/miniconda3/etc/profile.d/conda.sh
path = "../datasets/egohos/train/"

rgb_path = os.path.join(path, "image")
rgb = plt.imread(rgb_path+"/ego4d_0a02a1ed-a327-4753-b270-e95298984b96_11700.jpg")

seg_path = os.path.join(path, "label")
seg = plt.imread(seg_path+"/ego4d_0a02a1ed-a327-4753-b270-e95298984b96_11700.png")

print(seg.shape)

print(np.unique(seg))

# 0 bg
# 1, 2 hands
# 3,4,5,6,7,8 object

# print(list(seg[:, :, 1]))