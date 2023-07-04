import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, utils
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

from freihand_dataloader import FreiHand
from egohands_dataloader import EgoHands

import wandb

wandb.login()

torch.cuda.manual_seed(123)



class HandDataset(Dataset):
    """Hand Segmentation dataset."""

    def __init__(self, path = "../datasets/union/hand/", transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = 2
        self.rgb_path = os.path.join(path, "rgb")
        self.seg_path = os.path.join(path, "mask")
        self.data = []
        for filename in os.listdir(self.rgb_path):
            if filename.lower().endswith(('.jpg')):
                self.data.append(filename)

        if transform is None:
            transform_list = [
                            transforms.ToTensor(),
                            transforms.Resize(size=(256, 256), antialias=True),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            self.transform_rgb = transforms.Compose(transform_list)
            self.transform_mask = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize(size=(256, 256), antialias=True)
                                  ])
            # self.transform_mask = transforms.Compose([
            #                           transforms.ToTensor(),
            #                           transforms.Resize(size=(256, 256), antialias=True),
            #                           transforms.Lambda(lambda x: torch.eye(2)[x.long().squeeze()])
            #                       ])
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
            seg_file = os.path.join(self.seg_path, img_file)
            rgb = plt.imread(rgb_file)
            rgb = rgb/255

            mask = plt.imread(seg_file)
            mask = mask[:,:,0]
            mask = mask >=150
            mask = mask.astype(np.uint8)
            mask = np.eye(2)[mask]
            mask = self.transform_mask(mask)
            rgb = self.transform_rgb(rgb.astype(np.float32))
            
            return rgb, mask
        except Exception as err:
            print("Error:", err)
            return None

class Args:
    train_ratio = 0.8
    val_ratio = 0.2 
    batch_size = 16
    test_batch_size = 32
    cuda = True
    threads = 4
    num_epochs = 1000
    learning_rate = 0.0001
    model_dir = "/root/data/hand_object_segmentation/deeplab/training_results"
    checkpoints_dir = "/root/data/hand_object_segmentation/deeplab/checkpoints"

def log_image_table(images, ground_truth, predicted):
    table = wandb.Table(columns=["image", "predicted", "ground_truth"])
    _, ground_truth = torch.max(ground_truth, dim = 1)
    _, predicted = torch.max(predicted, dim = 1)

    for img, gt, pred in zip(images.to("cpu"), ground_truth.to("cpu"), predicted.to("cpu")):
        img = np.transpose(img.numpy(), (1, 2, 0))*255
        gt = gt.numpy()*255
        pred = pred.numpy()*255
        table.add_data(wandb.Image(img), wandb.Image(pred), wandb.Image(gt))
    wandb.log({"predictions_table":table}, commit=False)

def multi_acc(pred, label):
    _, tags = torch.max(pred, dim = 1)
    _, label = torch.max(label, dim = 1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc

def custom_DeepLabv3(out_channel):
  # model = deeplabv3_resnet101(pretrained=True, progress=True)
  model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
  model.classifier = DeepLabHead(2048, out_channel)
  return model

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    opt = Args()
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    wandb.init(
      # Set the project where this run will be logged
      project="Hand-Segmentation", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"DeepLabv3 Fine tune on hand segmentation", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": opt.learning_rate,
      "architecture": "DeepLabv3",
      "dataset": "Custom",
      "epochs": opt.num_epochs,
      })

    dataset = HandDataset()
    num_samples = len(dataset)
    train_size = int(opt.train_ratio * num_samples)
    val_size = num_samples - train_size
    n_steps_per_epoch = math.ceil(train_size / opt.batch_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=collate_fn, batch_size=opt.test_batch_size, shuffle=True)

    model = custom_DeepLabv3(out_channel = 2).to(device) 
    params = add_weight_decay(model, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr= opt.learning_rate)
    loss_fn = nn.CrossEntropyLoss().to(device) 

    epoch_losses_train = []
    epoch_losses_val = []
    for epoch in range(opt.num_epochs):
        print ("###########################")
        print ("Epoch: %d/%d" % (epoch+1, opt.num_epochs))

        model.train()
        batch_losses = []
        batch_acc = []
        for step, (imgs, label_imgs) in enumerate(train_loader):
            imgs = imgs.to(device) 
            label_imgs = label_imgs.to(device) 
            outputs = model(imgs) 
            outputs = outputs["out"]
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)
            acc = multi_acc(outputs, label_imgs).cpu().numpy()
            batch_acc.append(acc)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            metrics = {"train/train_loss_step": loss, 
                       "train/train_acc_step": acc,
                       "train/step": step}
            wandb.log(metrics)
            break

        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        epoch_losses_train.append(epoch_loss)
        
        train_metrics = {"train/train_loss": epoch_loss, 
                          "train/train_acc" : epoch_acc,
                          "train/epoch": epoch}

        print (f"Train Loss: {epoch_loss}, Train Acc: {epoch_acc}")
        # plt.figure(1)
        # plt.plot(epoch_losses_train, "k^")
        # plt.ylabel("loss")
        # plt.xlabel("epoch")
        # plt.title("Train loss per epoch")
        # plt.savefig("%s/epoch_losses_train.png" % opt.model_dir)

        model.eval()
        batch_losses = []
        batch_acc = []
        for step, (imgs, label_imgs) in enumerate(val_loader):
            with torch.no_grad():
                imgs = imgs.to(device) 
                label_imgs = label_imgs.to(device) 
                outputs = model(imgs)
                outputs = outputs["out"]
                loss = loss_fn(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                acc = multi_acc(outputs, label_imgs).cpu().numpy()
                batch_losses.append(loss_value)
                batch_acc.append(acc)
                metrics = {"val/val_loss_step": loss, 
                       "val/val_acc_step": acc,
                       "val/step": step}
                wandb.log(metrics)
        log_image_table(imgs, label_imgs, outputs)
        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        val_metrics = {"val/val_loss": epoch_loss, 
                       "val/val_accuracy": epoch_acc,
                       "val/epoch": epoch}
        wandb.log({**train_metrics, **val_metrics})
        epoch_losses_val.append(epoch_loss)
        print (f"Val Loss: {epoch_loss}, Val Acc: {epoch_acc}")
        # plt.figure(1)
        # plt.plot(epoch_losses_val, "k^")
        # plt.ylabel("loss")
        # plt.xlabel("epoch")
        # plt.title("Val loss per epoch")
        # plt.savefig("%s/epoch_losses_val.png" % opt.model_dir)
        # plt.close(1)

        if epoch%10 == 0:
            checkpoint_path = opt.checkpoints_dir + "/hand_only_" +"_epoch_" + str(epoch+1) + ".pth"
            torch.save(model.state_dict(), checkpoint_path)