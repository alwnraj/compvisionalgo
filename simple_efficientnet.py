# filename: simple_efficientnet.py
import os
import numpy as np
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

class TUM_RGBD_Dataset_RGB_Only(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        
        self.rgb_dict = read_file_list(os.path.join(base_dir, 'rgb.txt'))
        self.groundtruth = pd.read_csv(os.path.join(base_dir, 'groundtruth.txt'),
                                      sep=' ', comment='#', header=None,
                                      names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        
        self.timestamps = sorted(list(self.rgb_dict.keys()))

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        timestamp = self.timestamps[idx]
        
        # Get RGB image
        rgb_path = os.path.join(self.base_dir, self.rgb_dict[timestamp][0])
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        if self.transform:
            rgb_img = self.transform(rgb_img)
        
        # Get ground truth pose
        closest_gt = self.groundtruth.iloc[(self.groundtruth['timestamp'] - timestamp).abs().argsort()[0]]
        pose = closest_gt[['tx', 'ty', 'tz']].values.astype(np.float32)
        
        return rgb_img, torch.tensor(pose)

class EfficientNetRGBOnly(nn.Module):
    def __init__(self):
        super(EfficientNetRGBOnly, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Linear(1280, 3)
        
        # Time tracking
        self.forward_time = 0
        self.forward_count = 0

    def forward(self, rgb):
        start_time = time.time()
        features = self.efficientnet.features(rgb)
        features = self.efficientnet.avgpool(features)
        features = features.flatten(start_dim=1)
        pose = self.efficientnet.classifier(features)
        
        end_time = time.time()
        self.forward_time += (end_time - start_time)
        self.forward_count += 1
        
        return pose

class SimpleSLAM:
    def __init__(self):
        self.model = EfficientNetRGBOnly().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.trajectory = []
        self.frame_times = []
        
        # Additional timing metrics
        self.data_loading_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []

    def update(self, rgb, gt_pose):
        frame_start_time = time.time()
        
        # Optimizer step timing
        optimizer_start = time.time()
        self.optimizer.zero_grad()
        optimizer_time = time.time() - optimizer_start
        self.optimizer_times.append(optimizer_time)
        
        # Forward pass timing
        forward_start = time.time()
        estimated_pose = self.model(rgb)
        forward_time = time.time() - forward_start
        self.forward_times.append(forward_time)
        
        # Backward pass timing
        backward_start = time.time()
        loss = self.criterion(estimated_pose, gt_pose)
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.backward_times.append(backward_time)
        
        current_position = gt_pose.detach().cpu().numpy()
        self.trajectory.append(current_position)
        
        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        self.frame_times.append(frame_duration)
        
        return loss.item(), frame_duration, forward_time, backward_time, optimizer_time

def main():
    base_dir = 'rgbd_dataset_freiburg1_xyz'
    output_file = 'output_simple_efficientnet.txt'
    
    write_to_file("RGB-Only SLAM Execution Log", output_file)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TUM_RGBD_Dataset_RGB_Only(base_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    slam = SimpleSLAM()
    
    total_start_time = time.time()
    
    for i, (rgb, gt_pose) in enumerate(dataloader):
        data_load_end_time = time.time()
        data_load_time = data_load_end_time - (total_start_time if i == 0 else prev_iter_end)
        slam.data_loading_times.append(data_load_time)
        
        rgb, gt_pose = rgb.to(device), gt_pose.to(device)
        loss, frame_time, forward_time, backward_time, optimizer_time = slam.update(rgb, gt_pose)
        
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