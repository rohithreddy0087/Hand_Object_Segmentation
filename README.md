# Hand and Object Segmentation using DeepLabv3
Segmentation of hand and object for 3D hand pose estimation tasks

## Overview

This project focuses on the segmentation of hands and objects in images, a crucial task for various computer vision applications. Due to the lack of a dedicated dataset for this specific task, we adopted a novel approach. We leveraged hand pose estimation and existing segmentation datasets, including FreiHand, HO3D, EgoHOS, and EgoHands, to create a comprehensive dataset for training.

## Dataset

- **FreiHand**: Provides hand segmentation masks.
- **HO3D and EgoHOS**: Offers both hand and object masks.
- **EgoHands**: Contains hand segmentation masks.

By combining these datasets, we generated a diverse dataset for training our segmentation model. While FreiHand and EgoHands datasets provided hand segmentation masks, HO3D and EgoHOS datasets contributed both hand and object masks.

## Model

We employed the powerful DeepLabv3 model with a ResNet backbone pre-trained on the COCO dataset. The initial training phase focused solely on hand segmentation to ensure high accuracy. This initial training was conducted for a limited number of epochs.

## Training Strategy

Subsequently, we fine-tuned the model on the combined hand-object dataset. We modified the last layer to output two masks (hand and object), and the model underwent extended training for multiple epochs. This strategy resulted in a robust model that could effectively segment both hands and objects.

## Training Scalability

To handle the substantial dataset, our training script was implemented using PyTorch's Distributed Data Parallel (DDP), allowing us to utilize multiple GPUs efficiently. With over 100,000 training examples, this scalability was essential.

## Application

The primary goal of this project is to use the trained hand-object segmentation model to automatically annotate ground truths for 3D hand pose estimation tasks. This application aligns with the innovative HOnnotate method, enhancing the efficiency and accuracy of hand pose estimation.

We hope that this project's approach to creating a novel dataset and training a robust segmentation model can be beneficial for various computer vision tasks, especially those requiring precise hand-object segmentation. Feel free to explore the code and datasets included in this repository to adapt and extend this work for your specific needs.
