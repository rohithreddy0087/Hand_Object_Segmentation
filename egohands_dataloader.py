import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.io
import cv2
import random

#source ~/miniconda3/etc/profile.d/conda.sh
class EgoHands:
    """Class to process ego hands dataset 
    """
    def __init__(self, 
                 path = "../datasets/egohands/_LABELLED_SAMPLES/",
                 mat_path = "../datasets/egohands/metadata.mat",
                 save_path = "../datasets/union/hand/", 
                 ratio = 0.6):
        
        self.path = path
        self.num_classes = 2

        mat = scipy.io.loadmat(mat_path)
        videos = mat['video'][0]

        self.rgb_path = os.path.join(path, "rgb")
        self.seg_path = os.path.join(path, "mask")

        self.rgb_save_path = os.path.join(save_path, "rgb")
        self.seg_save_path = os.path.join(save_path, "mask")

        self.data = {}
        for vid in range(videos.shape[0]):
            vid_id = videos[vid][0][0]
            labelled_frames = videos[vid][6][0]
            for label in labelled_frames:
                frame_id = int(label[0])
                rgb_path = os.path.join(path, vid_id, f"frame_{frame_id:04d}.jpg")
                # print(rgb_path)
                # print(label)
                self.data[rgb_path] = [label[1], label[2], label[3], label[4] ]
        self.len = len(self.data)
        print("Total samples in Egohands dataset: ", self.len)
        indices = self.data.keys()
        self.selected_indices = random.sample(indices, int(ratio* self.len))
        self.index = 0
        
    def get_sample(self):
        path = self.selected_indices[self.index]
        polygons = self.data[path]
        rgb = plt.imread(path)
        shape = rgb.shape
        mask = np.zeros((shape[0], shape[1]), dtype = np.uint8)
        for coords in polygons:
            if coords.size == 0:
                continue
            coords = np.array(coords).astype(np.int32)
            coords = coords.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [coords], color=(255,255))

        mask = mask.astype(np.uint8)
        # one_hot_array = np.eye(self.num_classes)[mask]
        # print(one_hot_array.shape)
        rgb_save = os.path.join(self.rgb_save_path, f"ego_{self.index}.jpg")
        plt.imsave(rgb_save, rgb)

        seg_save = os.path.join(self.seg_save_path, f"ego_{self.index}.jpg")
        plt.imsave(seg_save, mask, cmap='gray')
        self.index += 1

if __name__ == "__main__":
    ego = EgoHands()
    for i in range(3500):
        ego.get_sample()
        # ego.get_sample()