# filename: vslam.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from feature_extract import TUM_RGBD_Dataset
from torchvision import transforms # Import the transforms module


class EfficientNetVSLAM(nn.Module):
    """
    VSLAM model using EfficientNet-B0 as the feature extractor.
    This model is designed to be compatible with features extracted 
    using the FeatureExtractor from feature_extract.py.
    """
    def __init__(self):
        super(EfficientNetVSLAM, self).__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Modify first conv layer to accept 6 channels (RGB + Depth)
        original_layer = efficientnet.features[0][0]
        efficientnet.features[0][0] = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            efficientnet.features[0][0].weight[:, :3] = original_layer.weight
            efficientnet.features[0][0].weight[:, 3:] = original_layer.weight

        # Use EfficientNet as feature extractor 
        self.feature_extractor = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

        # Pose estimation head (adjust input size)
        self.pose_head = nn.Sequential(
            nn.Flatten(start_dim=1),                
            nn.Linear(1280, 512),                 
            nn.ReLU(),
            nn.Linear(512, 3)  
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.avgpool(features)           
        features = torch.flatten(features, 1)       
        pose = self.pose_head(features)
        return pose, features

class VSLAM:
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        
        self.map = {}
        self.trajectory = []
        self.loop_closures = []
        self.map_features = []  # Store features in a list
        self.map_poses = []    # Store poses in a list

    def process_frame(self, features, gt_pose=None, index=None): 
        self.model.train()
        self.optimizer.zero_grad()

        estimated_pose = self.model.pose_head(features)
        # No need to squeeze here: estimated_pose = estimated_pose.squeeze(0)

        if gt_pose is not None:
            loss = self.criterion(estimated_pose, gt_pose)
            loss.backward()
            self.optimizer.step()
        
        # Update map and trajectory (use index as key)
        pose = estimated_pose.detach().cpu().numpy().squeeze()
        self.trajectory.append(pose)
        self.map_features.append(features.detach().cpu().numpy().squeeze())
        self.map_poses.append(pose) 
        
        # Perform loop closure detection 
        self._detect_loop_closure(features, pose)
        
        return estimated_pose, features

    def _detect_loop_closure(self, current_features, current_pose):
        if len(self.map) < 10:
            return
        
        # Simple loop closure based on feature similarity
        for pose, features in self.map.items():
            similarity = np.dot(current_features.cpu().numpy().squeeze(), features)
            if similarity > 0.9:  # Threshold for loop closure
                pose = np.array(pose)
                if np.linalg.norm(pose - current_pose) > 1.0:  # Distance threshold
                    self.loop_closures.append((tuple(pose), tuple(current_pose)))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # 1. Load pre-extracted features
    features_path = 'features.pt'
    features = torch.load(features_path, weights_only=True).to(device)

    # 2. Create the VSLAM model instance
    model = EfficientNetVSLAM().to(device)  

    # 3. Create VSLAM instance
    vslam = VSLAM(model)

    # Load your TUM RGBD dataset 
    transform = transforms.Compose([ # Now you can use transforms.Compose
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add normalization if needed
    ])
    dataset = TUM_RGBD_Dataset(base_dir='rgbd_dataset_freiburg1_xyz', transform=transform)

    # 4. Process features in VSLAM
    for i in range(len(dataset)): # Iterate through the dataset length 
        feature_vector = features[i].unsqueeze(0) 

        # Get the corresponding timestamp for the feature vector
        timestamp = dataset.synced_timestamps[i][0]  # Assuming (rgb_time, depth_time) in synced_timestamps

        # Find the closest ground truth pose based on the timestamp
        closest_gt = dataset.groundtruth.iloc[(dataset.groundtruth['timestamp'] - timestamp).abs().argsort()[0]]
        gt_pose = torch.tensor(closest_gt[['tx', 'ty', 'tz']].values.astype(np.float32)).to(device)

        estimated_pose, _ = vslam.process_frame(feature_vector, gt_pose)
        print(f"Estimated pose for feature {i}: {estimated_pose}") 

    print(f"Number of mapped points: {len(vslam.map_features)}")

if __name__ == "__main__":
    main()


''' what the results mean:
    You are seeing the estimated_pose for each feature vector. 
    Each pose is a 3-dimensional tensor representing the (x, y, z) position of the camera (or sensor) as 
    estimated by the VSLAM system.

    grad_fn=<SqueezeBackward1>: This just indicates that the tensor is still part of the computational 
    graph and has a gradient function associated with it (because you are calling model.train()).

'''