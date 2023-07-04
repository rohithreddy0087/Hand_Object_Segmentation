import matplotlib.pyplot as plt
import os
import numpy as np
import random

#source ~/miniconda3/etc/profile.d/conda.sh

class FreiHand:
    """Class to process freihand dataset 
    """
    def __init__(self, path = "../datasets/freihand/training/", save_path = "../datasets/union/hand/", ratio = 0.4):
        
        self.path = path
        self.num_classes = 2
        self.rgb_path = os.path.join(path, "rgb")
        self.seg_path = os.path.join(path, "mask")

        self.rgb_save_path = os.path.join(save_path, "rgb")
        self.seg_save_path = os.path.join(save_path, "mask")

        self.data = []
        list_dir = os.listdir(self.seg_path)
        for rgb in list_dir:
            self.data.append(rgb)
        self.len = len(self.data)
        print("Total samples in FreiHand dataset: ", self.len)
        indices = list(range(self.len))
        self.selected_indices = random.sample(indices, int(ratio* self.len))
        self.index = 0
        
    def get_sample(self):
        path = self.selected_indices[self.index]
        img_name = self.data[path]
        rgb = plt.imread(os.path.join(self.rgb_path , img_name))
        seg = plt.imread(os.path.join(self.seg_path , img_name))
        seg = seg[:,:,0] >= 150
        seg = seg.astype(np.uint8)*255


        rgb_save = os.path.join(self.rgb_save_path, f"frei_{self.index}.jpg")
        plt.imsave(rgb_save, rgb)

        seg_save = os.path.join(self.seg_save_path, f"frei_{self.index}.jpg")
        plt.imsave(seg_save, seg)
        self.index += 1

if __name__ == "__main__":
    frei = FreiHand()
    for i in range(13000):
        frei.get_sample()
# 0 bg
# 1, 2 hands
# 3,4,5,6,7,8 object