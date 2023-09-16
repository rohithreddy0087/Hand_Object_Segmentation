import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from torchvision import transforms, utils
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
from torchvision.models.segmentation import deeplabv3_resnet101

torch.cuda.manual_seed(123)



class InferenceDataset(Dataset):
    """Hand Segmentation dataset."""

    def __init__(self, path = "/root/data/frames/20230704_154715", transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rgb_path = path
        self.data = []
        for i, filename in enumerate(os.listdir(self.rgb_path)):
            if filename.lower().endswith(('.png')):
                if i%5 == 0:
                    self.data.append(filename)

        if transform is None:
            transform_list = [
                            transforms.ToTensor(),
                            transforms.Resize(size=(256, 256), antialias=True)]
            self.transform_rgb = transforms.Compose(transform_list)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_file = self.data[idx]
            rgb_file = os.path.join(self.rgb_path, img_file)
            rgb = plt.imread(rgb_file)
            rgb = self.transform_rgb(rgb.astype(np.float32))
            return rgb
        except Exception as err:
            print("Error:", err)
            return None

class Args:
    cuda = True
    epoch = 100
    checkpoints_dir = "/root/data/hand_object_segmentation/deeplab/checkpoints"
    checkpoint_path = checkpoints_dir + "/hand_only_" +"_epoch_" + str(epoch) + ".pth"
    save_path = "/root/data/hand_object_segmentation/deeplab/results/hands"

def multi_acc(pred, label):
    _, tags = torch.max(pred, dim = 1)
    _, label = torch.max(label, dim = 1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc

def custom_DeepLabv3(out_channel):
  model = deeplabv3_resnet101()
  model.classifier = DeepLabHead(2048, out_channel)
  model.aux_classifier = FCNHead(1024, out_channel)
  return model

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    opt = Args()
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    dataset = InferenceDataset()
    test_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)

    model = custom_DeepLabv3(out_channel = 2).to(device) 
    model.eval()
    model_wts = torch.load(opt.checkpoint_path)
    model.load_state_dict(model_wts)
    model.aux_classifier = None
    batch_losses = []
    batch_acc = []
    for step, (imgs) in enumerate(test_loader):
        print(step)
        with torch.no_grad():
            imgs = imgs.to(device) 
            outputs = model(imgs)
            outputs = outputs["out"][0]
            outputs = outputs.argmax(0)

            imgs = imgs.to("cpu").numpy()
            out = outputs.to("cpu").numpy()
            img = np.transpose(imgs[0], (1, 2, 0))
            out = out*255
            plt.figure()
            plt.subplot(1,2,1)
            plt.axis("off")
            plt.title("Input Images")
            plt.imshow(img)

            # Plot the fake images from the last epoch
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Segmented Images")
            plt.imshow(out)

            plt.savefig(opt.save_path+"/"+str(step)+".png")
            plt.close()
