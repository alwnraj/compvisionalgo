# filename: efficientnetalgo.py
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

def write_to_file(text, file):
    with open(file, 'w') as f:
        f.write(text + '\n')

def append_to_file(text, file):
    with open(file, 'a') as f:
        f.write(text + '\n')

def read_file_list(filename):
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

        self.rgb_timestamps = list(self.rgb_dict.keys())
        self.depth_timestamps = list(self.depth_dict.keys())
        self.synced_timestamps = self.synchronize_timestamps()

    def synchronize_timestamps(self):
        synced = []
        for rgb_time in self.rgb_timestamps:
            depth_time = min(self.depth_timestamps, key=lambda x: abs(x - rgb_time))
            if abs(rgb_time - depth_time) < 0.02:
                synced.append((rgb_time, depth_time))
        return synced

    def __len__(self):
        return len(self.synced_timestamps)

    def __getitem__(self, idx):
        rgb_time, depth_time = self.synced_timestamps[idx]

        rgb_path = os.path.join(self.base_dir, self.rgb_dict[rgb_time][0])
        depth_path = os.path.join(self.base_dir, self.depth_dict[depth_time][0])

        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('RGB')

        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img)

        closest_gt = self.groundtruth.iloc[(self.groundtruth['timestamp'] - rgb_time).abs().argsort()[0]]
        pose = closest_gt[['tx', 'ty', 'tz']].values.astype(np.float32)

        return rgb_img, depth_img, torch.tensor(pose)

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Modify first layer to accept 6 channels (RGB + Depth)
        original_layer = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Initializing the new layer with weights from the pretrained model
        with torch.no_grad():
            new_weights = torch.zeros_like(self.efficientnet.features[0][0].weight)
            new_weights[:, :3, :, :] = original_layer.weight
            new_weights[:, 3:, :, :] = original_layer.weight
            self.efficientnet.features[0][0].weight = nn.Parameter(new_weights)
        
        self.efficientnet.classifier = nn.Linear(1280, 3)
        
        # Time tracking
        self.forward_time = 0
        self.forward_count = 0

    def forward(self, x):
        start_time = time.time()
        features = self.efficientnet.features(x)
        features = self.efficientnet.avgpool(features)
        features = features.flatten(start_dim=1)
        pose = self.efficientnet.classifier(features)
        
        end_time = time.time()
        self.forward_time += (end_time - start_time)
        self.forward_count += 1
        
        return pose, features

class SLAM:
    def __init__(self):
        self.feature_extractor = EfficientNetFeatureExtractor().to(device)
        self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.map = {}
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.trajectory = [self.current_position]
        
        # Timing metrics
        self.frame_times = []
        self.data_loading_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []

    def update(self, rgb, depth, gt_pose):
        frame_start_time = time.time()
        
        # Optimizer step timing
        optimizer_start = time.time()
        self.optimizer.zero_grad()
        optimizer_time = time.time() - optimizer_start
        self.optimizer_times.append(optimizer_time)
        
        # Forward pass timing
        forward_start = time.time()
        x = torch.cat((rgb, depth), dim=1)
        estimated_pose, features = self.feature_extractor(x)
        forward_time = time.time() - forward_start
        self.forward_times.append(forward_time)
        
        # Backward pass timing
        backward_start = time.time()
        loss = self.criterion(estimated_pose, gt_pose.float())
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.backward_times.append(backward_time)

        self.current_position = gt_pose.detach().cpu().numpy().squeeze()
        self.trajectory.append(self.current_position)
        self.map[tuple(self.current_position)] = features.detach().cpu().numpy().squeeze()

        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        self.frame_times.append(frame_duration)
        
        return loss.item(), frame_duration, forward_time, backward_time, optimizer_time

def main():
    base_dir = 'rgbd_dataset_freiburg1_xyz'
    output_file = 'output_efficientnetb0.txt'
    #output_file = 'original_efficientnetb0.txt' 

    write_to_file("EfficientNet-B0 SLAM (RGB+Depth) Execution Log", output_file)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TUM_RGBD_Dataset(base_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    slam = SLAM()

    total_start_time = time.time()
    prev_iter_end = total_start_time

    for i, (rgb, depth, gt_pose) in enumerate(dataloader):
        data_load_end_time = time.time()
        data_load_time = data_load_end_time - prev_iter_end
        slam.data_loading_times.append(data_load_time)
        
        rgb, depth, gt_pose = rgb.to(device), depth.to(device), gt_pose.to(device)
        loss, frame_time, forward_time, backward_time, optimizer_time = slam.update(rgb, depth, gt_pose)

        if i % 10 == 0:
            current_time = time.time() - total_start_time
            avg_frame_time = np.mean(slam.frame_times[-10:]) if slam.frame_times else 0
            append_to_file(f"Frame {i}, Loss: {loss:.4f}, Frame Time: {frame_time:.4f}s, Running Time: {current_time:.2f}s, Avg Frame Time: {avg_frame_time:.4f}s", output_file)
        
        prev_iter_end = time.time()

    total_time = time.time() - total_start_time

    # Final statistics
    append_to_file("\nFinal Stats:", output_file)
    append_to_file(f"Total runtime: {total_time:.2f} seconds", output_file)
    append_to_file(f"Frames processed: {len(dataset)}", output_file)
    append_to_file(f"Average frame processing time: {np.mean(slam.frame_times):.4f} seconds", output_file)
    append_to_file(f"Average forward pass time: {np.mean(slam.forward_times):.4f} seconds", output_file)
    append_to_file(f"Average backward pass time: {np.mean(slam.backward_times):.4f} seconds", output_file)
    append_to_file(f"Average optimizer time: {np.mean(slam.optimizer_times):.4f} seconds", output_file)
    append_to_file(f"Average data loading time: {np.mean(slam.data_loading_times):.4f} seconds", output_file)

if __name__ == "__main__":
    main()