import os
import time
import torch
import threading
import queue
from vslam import EfficientNetVSLAM, VSLAM
from feature_extract import FeatureExtractor, TUM_RGBD_Dataset
from object_classifier import ObjectClassifier
from torchvision import transforms

class ParallelProcessor:
    def __init__(self, dataset_path, num_classes=10, feature_dim=1280, feature_queue_size=5):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = FeatureExtractor().to(self.device)
        self.object_classifier = ObjectClassifier(num_classes=num_classes, feature_dim=feature_dim).to(self.device)
        self.vslam_model = EfficientNetVSLAM().to(self.device)
        self.vslam = VSLAM(self.vslam_model)

        self.feature_queue = queue.Queue(maxsize=feature_queue_size)
        self.classification_queue = queue.Queue()
        self.vslam_queue = queue.Queue()

    def run(self):
        start_time = time.time()
        transform = self.get_transform()
        dataset = TUM_RGBD_Dataset(self.dataset_path, transform=transform)

        for idx in range(len(dataset)):
            frame_start = time.time()
            rgb, depth, _ = dataset[idx]
            rgb = rgb.unsqueeze(0).to(self.device)
            depth = depth.unsqueeze(0).to(self.device)
            rgbd_image = torch.cat((rgb, depth), dim=1)
            features = self.feature_extractor(rgbd_image)

            classification_thread = threading.Thread(
                target=self.process_classification, args=(features, idx)
            )
            vslam_thread = threading.Thread(
                target=self.process_vslam, args=(features, idx)
            )

            classification_thread.start()
            vslam_thread.start()

            classification_thread.join()
            vslam_thread.join()

            print(f"Processed frame {idx}")

            frame_end = time.time()
            print(f"Frame {idx} processed in: {frame_end - frame_start:.4f} seconds")

        end_time = time.time()
        print(f"Total runtime: {end_time - start_time:.4f} seconds")

    def process_classification(self, features, idx):
        self.object_classifier.eval()
        with torch.no_grad():
            outputs = self.object_classifier(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            self.classification_queue.put((probabilities, idx))
            print(f"Object classification: Processed features for image {idx}")

        del features
        torch.cuda.empty_cache()

    def process_vslam(self, features, idx):
        dataset = TUM_RGBD_Dataset(self.dataset_path, transform=self.get_transform())
        _, _, gt_pose = dataset[idx]
        gt_pose = gt_pose.unsqueeze(0).to(self.device)
        estimated_pose, _ = self.vslam.process_frame(features, gt_pose, idx)
        self.vslam_queue.put((estimated_pose, idx))
        print(f"VSLAM: Processed frame {idx}")

        del features
        torch.cuda.empty_cache()

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

if __name__ == "__main__":
    dataset_path = 'rgbd_dataset_freiburg1_xyz'
    processor = ParallelProcessor(dataset_path)
    processor.run()
