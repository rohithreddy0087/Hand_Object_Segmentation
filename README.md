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

# Project Results
## Hand only Segmentation
![combined_4](/results/combined_4.png)
![combined_5](/results/combined_5.png)
![combined_6](/results/combined_6.png)

## Hand-Object Segmentation
![combined_1](/results/combined_1.png)
![combined_2](/results/combined_2.png)
![combined_3](/results/combined_3.png)

## Preliminary Results of 3D Hand-Pose Estimation
![combined_7](/results/combined_7.png)
![combined_8](/results/combined_8.png)

## Model Performance

After extensive training and evaluation, the segmentation model yielded impressive results, indicating its effectiveness in accurately segmenting hands and objects in images. The following metrics were obtained:

- **Train Jaccard Index**: 0.94
- **Train Accuracy**: 98.8%
- **Validation Jaccard Index**: 0.90
- **Validation Accuracy**: 97.4%

## Metric Choice

It's worth noting that accuracy, while often used as a performance metric, can saturate quickly in segmentation tasks. Therefore, we adopted the **Jaccard Index** as a more reliable evaluation metric for our segmentation model. This metric provides a robust measure of the model's segmentation accuracy, taking into account both true positives and false positives while penalizing false negatives, making it well-suited for our specific task.

## Application

The primary goal of this project is to use the trained hand-object segmentation model to automatically annotate ground truths for 3D hand pose estimation tasks. This application aligns with the innovative HOnnotate method, enhancing the efficiency and accuracy of hand pose estimation.

We hope that this project's approach to creating a novel dataset and training a robust segmentation model can be beneficial for various computer vision tasks, especially those requiring precise hand-object segmentation. Feel free to explore the code and datasets included in this repository to adapt and extend this work for your specific needs.
