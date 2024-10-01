#filename: efficientalgo.py

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import pandas as pd
from PIL import Image
import time

device = torch.device("cpu")

# Modify the print function to write to a file
def write_to_file(text, file):
    with open(file, 'a') as f:
        f.write(text + '\n')

def read_file_list(filename):
    """
    Reads a trajectory from a text file.
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)

class TUM_RGBD_Dataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform

        self.rgb_dict = read_file_list(os.path.join(base_dir, 'rgb.txt'))
        self.depth_dict = read_file_list(os.path.join(base_dir, 'depth.txt'))
        self.groundtruth = pd.read_csv(os.path.join(base_dir, 'groundtruth.txt'),
                                       sep=' ', comment='#', header=None,
                                       names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        # Synchronize RGB and depth images
        self.rgb_timestamps = list(self.rgb_dict.keys())
        self.depth_timestamps = list(self.depth_dict.keys())
        self.synced_timestamps = self.synchronize_timestamps()

    def synchronize_timestamps(self):
        synced = []
        for rgb_time in self.rgb_timestamps:
            depth_time = min(self.depth_timestamps, key=lambda x: abs(x - rgb_time))
            if abs(rgb_time - depth_time) < 0.02:  # 20ms threshold
                synced.append((rgb_time, depth_time))
        return synced

    def __len__(self):
        return len(self.synced_timestamps)

    def __getitem__(self, idx):
        rgb_time, depth_time = self.synced_timestamps[idx]

        rgb_path = os.path.join(self.base_dir, self.rgb_dict[rgb_time][0])
        depth_path = os.path.join(self.base_dir, self.depth_dict[depth_time][0])

        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('RGB')  # Convert depth to RGB

        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img)

        # Get the closest ground truth pose
        closest_gt = self.groundtruth.iloc[(self.groundtruth['timestamp'] - rgb_time).abs().argsort()[0]]
        pose = closest_gt[['tx', 'ty', 'tz']].values

        return rgb_img, depth_img, pose

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) # I'm using the up-to-date models since it is throwing errors at me
        self.efficientnet.classifier = nn.Identity()  # Remove the final classification layer

        # Modify the first convolutional layer to accept 6 channels
        self.efficientnet.features[0][0] = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.fc = nn.Linear(1280, 3)  # EfficientNet-B0 outputs 1280 features
        self.feature_extraction_time = 0
        self.feature_extraction_count = 0

    def forward(self, rgb, depth):
        start_time = time.time()
        x = torch.cat((rgb, depth), dim=1)  # Combine RGB and depth channels
        features = self.efficientnet(x)
        pose = self.fc(features)
        end_time = time.time()
        self.feature_extraction_time += (end_time - start_time)
        self.feature_extraction_count += 1
        return pose, features

class SLAM:
    def __init__(self):
        self.feature_extractor = EfficientNetFeatureExtractor().to(device)
        self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.map = {}
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.trajectory = [self.current_position]
        self.checkpoint_times = []

    def update(self, rgb, depth, gt_pose):
        start_time = time.time()
        self.optimizer.zero_grad()

        estimated_pose, features = self.feature_extractor(rgb, depth)

        loss = self.criterion(estimated_pose, gt_pose.float())
        loss.backward()
        self.optimizer.step()

        # Update position and map
        self.current_position = gt_pose.detach().cpu().numpy().squeeze()
        self.trajectory.append(self.current_position)
        self.map[tuple(self.current_position)] = features.detach().cpu().numpy().squeeze()

        end_time = time.time()
        self.checkpoint_times.append((start_time, end_time))

        return loss.item()

def main():
    base_dir = 'rgbd_dataset_freiburg1_xyz'
    output_file = 'output_efficientnetb0.txt'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet-B0 expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    dataset = TUM_RGBD_Dataset(base_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    slam = SLAM()

    total_start_time = time.time()

    for i, (rgb, depth, gt_pose) in enumerate(dataloader):
        rgb, depth, gt_pose = rgb.to(device), depth.to(device), gt_pose.to(device)

        loss = slam.update(rgb, depth, gt_pose)

        if i % 10 == 0:
            checkpoint_start, checkpoint_end = slam.checkpoint_times[-1]
            write_to_file(f"Frame {i}, Loss: {loss:.4f}, Position: {slam.current_position}", output_file)
            write_to_file(f"Checkpoint time: {checkpoint_end - checkpoint_start:.4f} seconds", output_file)

    total_end_time = time.time()

    write_to_file(f"SLAM completed. Total frames processed: {len(dataset)}", output_file)
    write_to_file(f"Total runtime: {total_end_time - total_start_time:.2f} seconds", output_file)

    avg_feature_extraction_time = slam.feature_extractor.feature_extraction_time / slam.feature_extractor.feature_extraction_count
    write_to_file(f"Average feature extraction time: {avg_feature_extraction_time:.4f} seconds", output_file)
    write_to_file(f"Total features extracted: {slam.feature_extractor.feature_extraction_count}", output_file)
    write_to_file(f"Total feature extraction time: {slam.feature_extractor.feature_extraction_time:.2f} seconds", output_file)

if __name__ == "__main__":
    main()
