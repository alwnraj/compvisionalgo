# filename: feature_extract.py
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import pandas as pd
import numpy as np
import subprocess 


def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)

class TUM_RGBD_Dataset(Dataset):
    """
    Loads and preprocesses RGB-D data from the TUM RGB-D dataset.

    Args:
        base_dir (str): Path to the dataset directory.
        transform (callable, optional): Transformations to apply to the images.
    """
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
        """Synchronizes RGB and depth timestamps."""
        synced = []
        for rgb_time in self.rgb_timestamps:
            depth_time = min(self.depth_timestamps, key=lambda x: abs(x - rgb_time))
            if abs(rgb_time - depth_time) < 0.02:
                synced.append((rgb_time, depth_time))
        return synced

    def __len__(self):
        """Returns the number of synchronized RGB-D pairs."""
        return len(self.synced_timestamps)

    def __getitem__(self, idx):
        """
        Returns a single RGB-D image pair and its pose.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: (rgb_img, depth_img, pose)
                - rgb_img: Transformed RGB image.
                - depth_img: Transformed depth image.
                - pose: Ground truth pose as a tensor.
        """
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
    
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Modify first layer to accept 6 channels (RGB + Depth)
        original_layer = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Initialize weights for the new channels (you can use better initialization if needed)
        with torch.no_grad():
            new_weights = torch.zeros_like(self.efficientnet.features[0][0].weight)
            new_weights[:, :3, :, :] = original_layer.weight
            new_weights[:, 3:, :, :] = original_layer.weight
            self.efficientnet.features[0][0].weight = nn.Parameter(new_weights)

        # Remove the classifier head
        self.efficientnet.classifier = nn.Identity()

        # Add AdaptiveAvgPool2d to get a 1x1 feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.efficientnet.features(x)
        features = self.adaptive_pool(features)
        features = features.flatten(start_dim=1)
        return features

def extract_features_and_save(dataset_path, output_path, batch_size=32, device='cpu'):
    """
    Extracts features from a dataset and saves them to disk.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add normalization if you're using a pre-trained model
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TUM_RGBD_Dataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    all_features = []
    with torch.no_grad():
        for batch_idx, (rgb, depth, _) in enumerate(dataloader):
            rgb = rgb.to(device)
            depth = depth.to(device)
            rgbd_image = torch.cat((rgb, depth), dim=1)
            features = feature_extractor(rgbd_image)
            all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=0)
    torch.save(all_features, output_path)


if __name__ == "__main__":
    base_dir = 'rgbd_dataset_freiburg1_xyz' 
    output_path = 'features.pt'

    # 1. Extract features
    extract_features_and_save(base_dir, output_path)

'''
    # 2. Run object_classifier.py using subprocess ( a branching to b)
    print("Feature extraction complete. Starting object classification...")
    subprocess.run(["python", "object_classifier.py"])

'''