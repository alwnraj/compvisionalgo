import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import time

device = torch.device("cpu")

# ... (previous code remains the same) ...
def read_file_list(filename):
    """
    Reads a trajectory from a text file.
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
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

        rgb_img = Image.open(rgb_path)
        depth_img = Image.open(depth_path)

        # Print image modes to debug
        print(f"RGB image mode: {rgb_img.mode}, Depth image mode: {depth_img.mode}")

        # Ensure depth image is in 'L' mode (grayscale)
        if depth_img.mode != 'L':
            print(f"Converting depth image from {depth_img.mode} to 'L'")
            depth_img = depth_img.convert('L')

        # Transform images
        if self.transform:
            rgb_img = self.transform(rgb_img)

            try:
                depth_img = self.transform(depth_img)
            except Exception as e:
                print(f"Error transforming depth image: {e}")
                raise

        # Get the closest ground truth pose
        closest_gt = self.groundtruth.iloc[(self.groundtruth['timestamp'] - rgb_time).abs().argsort()[0]]
        pose = closest_gt[['tx', 'ty', 'tz']].values

        return rgb_img, depth_img, pose


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 3)  # Output 3D pose (x, y, z)
        self.feature_extraction_time = 0
        self.feature_extraction_count = 0

    def forward(self, rgb, depth):
        start_time = time.time()
        x = torch.cat((rgb, depth), dim=1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        features = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(features))
        x = self.fc2(x)
        end_time = time.time()
        self.feature_extraction_time += (end_time - start_time)
        self.feature_extraction_count += 1
        return x, features

class SLAM:
    def __init__(self):
        self.feature_extractor = FeatureExtractor().to(device)
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

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
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
            print(f"Frame {i}, Loss: {loss:.4f}, Position: {slam.current_position}")
            print(f"Checkpoint time: {checkpoint_end - checkpoint_start:.4f} seconds")

    total_end_time = time.time()

    print("SLAM completed. Total frames processed:", len(dataset))
    print(f"Total runtime: {total_end_time - total_start_time:.2f} seconds")

    avg_feature_extraction_time = slam.feature_extractor.feature_extraction_time / slam.feature_extractor.feature_extraction_count
    print(f"Average feature extraction time: {avg_feature_extraction_time:.4f} seconds")
    print(f"Total features extracted: {slam.feature_extractor.feature_extraction_count}")
    print(f"Total feature extraction time: {slam.feature_extractor.feature_extraction_time:.2f} seconds")

if __name__ == "__main__":
    main()