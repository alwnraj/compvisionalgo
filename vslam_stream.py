#filename: vslam_stream.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from feature_extract_2 import TUM_RGBD_Dataset
from torchvision import transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class StreamingVSLAM:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

        self.map_features = []
        self.map_poses = []
        self.trajectory = []
        self.loop_closures = []

        # Initialize dataset for ground truth poses
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dataset = TUM_RGBD_Dataset('rgbd_dataset_freiburg1_xyz', transform=transform)

    def process(self, feature):
        """Process a single feature vector in real-time."""
        self.model.eval()

        # Move the feature to the specified device
        feature = feature.to(self.device)

        try:
            # Get the ground truth pose for this feature
            if len(self.map_features) < len(self.dataset):
                timestamp = self.dataset.synced_timestamps[len(self.map_features)][0]
                closest_gt = self.dataset.groundtruth.iloc[
                    (self.dataset.groundtruth['timestamp'] - timestamp).abs().argsort()[0]
                ]
                gt_pose = torch.tensor(
                    closest_gt[['tx', 'ty', 'tz']].values.astype(np.float32)
                ).to(self.device)
            else:
                gt_pose = torch.zeros(3, device=self.device)  # Default if no ground truth available

            # Estimate pose using the model
            estimated_pose = self.model.pose_head(feature.unsqueeze(0)).squeeze()

            # Calculate loss and backpropagate (optional for online adaptation)
            loss = self.criterion(estimated_pose, gt_pose)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Save results
            pose = estimated_pose.detach().cpu().numpy()
            self.trajectory.append(pose)
            self.map_features.append(feature.cpu().numpy())
            self.map_poses.append(pose)

            # Check for loop closures
            self._detect_loop_closure(feature, pose)

            return pose  # Return the estimated pose
        except Exception as e:
            logging.error(f"Error during VSLAM processing: {str(e)}")
            raise

    def _detect_loop_closure(self, current_features, current_pose):
        """Detect loop closures in the trajectory"""
        if len(self.map_features) < 10:
            return

        current_features = current_features.cpu().numpy().squeeze()
        for i, (pose, features) in enumerate(zip(self.map_poses[:-10], self.map_features[:-10])):
            similarity = np.dot(current_features, features)
            if similarity > 0.9:  # Threshold for loop closure
                if np.linalg.norm(pose - current_pose) > 1.0:  # Distance threshold
                    self.loop_closures.append((tuple(pose), tuple(current_pose)))
                    logging.info(f"Detected loop closure between {pose} and {current_pose}")

    def get_results(self):
        """Return current results"""
        return {
            'trajectory_length': len(self.trajectory),
            'loop_closures': len(self.loop_closures),
            'last_pose': self.trajectory[-1] if self.trajectory else None
        }

def main():
    """Process a single batch of features (for testing)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vslam = StreamingVSLAM(device=device)

    # Process a dummy batch
    dummy_features = torch.randn(32, 1280)
    try:
        results = vslam.process(dummy_features)
        logging.info(f"Processed batch, got {len(results)} poses")
    except Exception as e:
        logging.error(f"Error during VSLAM processing: {str(e)}")

if __name__ == "__main__":
    main()