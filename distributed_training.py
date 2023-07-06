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
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

import wandb
import argparse

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

torch.cuda.manual_seed(123)
os.environ["WANDB_API_KEY"] = "d21415063a4a04fd70bb4f7728a855e8d4c29ba2"
wandb.login()

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='simple distributed training job')
        self.parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio for training data (default: 0.8)')
        self.parser.add_argument('--val_ratio', type=float, default=0.3, help='Ratio for validation data (default: 0.2)')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Input batch size (default: 16)')
        self.parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size (default: 32)')
        self.parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
        self.parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loading (default: 4)')
        self.parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train (default: 1000)')
        self.parser.add_argument('--save_every', type=int, default=5, help='Save checkpoints for every 10 epochs')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
        self.parser.add_argument('--model_dir', type=str, default='/root/data/hand_object_segmentation/deeplab/training_results', help='Model directory (default: /root/data/hand_object_segmentation/deeplab/training_results)')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/root/data/hand_object_segmentation/deeplab/checkpoints', help='Checkpoints directory (default: /root/data/hand_object_segmentation/deeplab/checkpoints)')

    def parse_args(self):
        return self.parser.parse_args()

class HandDataset(Dataset):
    """Hand Segmentation dataset."""

    def __init__(self, path = "/root/data/hand_object_segmentation/datasets/union/hand/", transform=None):
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

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        wandb,
        opt,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.loss_fn = nn.CrossEntropyLoss()
        self.wandb = wandb
        self.opt = opt

    def log_image_table(self, images, ground_truth, predicted):
        table = wandb.Table(columns=["image", "predicted", "ground_truth"])
        _, ground_truth = torch.max(ground_truth, dim = 1)
        _, predicted = torch.max(predicted, dim = 1)

        for img, gt, pred in zip(images.to("cpu"), ground_truth.to("cpu"), predicted.to("cpu")):
            img = np.transpose(img.numpy(), (1, 2, 0))*255
            gt = gt.numpy()*255
            pred = pred.numpy()*255
            table.add_data(wandb.Image(img), wandb.Image(pred), wandb.Image(gt))
        self.wandb.log({"predictions_table":table}, commit=False)

    def _multi_acc(self, pred, label):
        _, tags = torch.max(pred, dim = 1)
        _, label = torch.max(label, dim = 1)
        corrects = (tags == label).float()
        acc = corrects.sum() / corrects.numel()
        acc = acc * 100
        return acc

    def _run_train_batch(self, source, targets, step, batch_losses, batch_acc):
        self.optimizer.zero_grad()
        outputs = self.model(source) 
        loss_out = self.loss_fn(outputs["out"], targets)
        loss_aux = self.loss_fn(outputs["aux"], targets)
        loss = loss_out + 0.1*loss_aux
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)
        acc = self._multi_acc(outputs["out"], targets).cpu().numpy()
        batch_acc.append(acc)
        loss.backward()
        self.optimizer.step()
        metrics = {"train/train_loss_step": loss, 
                       "train/train_acc_step": acc,
                       "train/step": step}
        self.wandb.log(metrics)

    def _run_val_batch(self, source, targets, step, batch_losses, batch_acc):
        outputs = self.model(source) 
        outputs = outputs["out"]
        loss = self.loss_fn(outputs, targets)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)
        acc = self._multi_acc(outputs, targets).cpu().numpy()
        batch_acc.append(acc)

        metrics = {"val/val_loss_step": loss, 
                    "val/val_acc_step": acc,
                    "val/step": step}
        self.wandb.log(metrics)
        if step in [int(len(self.val_data)/4), int(len(self.val_data)/2),\
                     int(3*len(self.val_data)/4), len(self.val_data)-2]:
            self.log_image_table(source, outputs, targets)

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        batch_losses = []
        batch_acc = []
        for step, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_train_batch(source, targets, step, batch_losses, batch_acc)

        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        
        train_metrics = {"train/train_loss": epoch_loss, 
                          "train/train_acc" : epoch_acc,
                          "train/epoch": epoch}

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Train Loss: {epoch_loss}, Train Acc: {epoch_acc}")

        self.model.eval()
        batch_losses = []
        batch_acc = []
        self.val_data.sampler.set_epoch(epoch)
        for step, (source, targets) in enumerate(self.val_data):
            with torch.no_grad():
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                self._run_val_batch(source, targets, step, batch_losses, batch_acc)        
        
        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        val_metrics = {"val/val_loss": epoch_loss, 
                       "val/val_accuracy": epoch_acc,
                       "val/epoch": epoch}
        self.wandb.log({**train_metrics, **val_metrics})
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Val Loss: {epoch_loss}, Val Acc: {epoch_acc}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        checkpoint_path = self.opt.checkpoints_dir + "/hand_only_" +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(ckp, checkpoint_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def custom_DeepLabv3(out_channel):
    # model = deeplabv3_resnet101(pretrained=True, progress=True)
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier = DeepLabHead(2048, out_channel)
    model.aux_classifier = FCNHead(1024, out_channel)
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

def load_train_objs(opt):
    dataset = HandDataset()
    num_samples = len(dataset)
    train_size = int(opt.train_ratio * num_samples)
    val_size = num_samples - train_size
    n_steps_per_epoch = math.ceil(train_size / opt.batch_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = custom_DeepLabv3(out_channel = 2)
    params = add_weight_decay(model, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr= opt.learning_rate)

    return train_dataset, val_dataset, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, opt, wandb):
    ddp_setup(rank, world_size)
    train_dataset, val_dataset, model, optimizer = load_train_objs(opt)
    train_data = prepare_dataloader(train_dataset, opt.batch_size)
    val_data = prepare_dataloader(val_dataset, opt.test_batch_size)
    trainer = Trainer(model, train_data, val_data, optimizer, rank, wandb, opt, opt.save_every)
    trainer.train(opt.num_epochs)
    destroy_process_group()


if __name__ == "__main__":
    opt = Args().parse_args()
    run_wandb = wandb.init(
      # Set the project where this run will be logged
      project="Hand-Segmentation", 
      group="DDP", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"DeepLabv3 Fine tune on hand segmentation", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": opt.learning_rate,
      "architecture": "DeepLabv3",
      "dataset": "Custom",
      "epochs": opt.num_epochs,
      })
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, opt, run_wandb), nprocs=world_size)
