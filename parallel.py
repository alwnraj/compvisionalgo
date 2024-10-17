
# this code uses runs object_classifier.py and vslam.py in parallel using the output from features_extract.py

import threading
import torch
import time
from queue import Queue
import subprocess
import os
import sys
from vslam import main as vslam_main
from object_classifier import main as classifier_main
from feature_extract import extract_features_and_save

class FeatureProcessor:
    def __init__(self):
        self.feature_path = 'features.pt'
        self.feature_ready = threading.Event()
        self.completion_queue = Queue()

    def extract_features(self):
        """Extract features and notify threads when ready"""
        print("Starting feature extraction...")
        base_dir = 'rgbd_dataset_freiburg1_xyz'
        extract_features_and_save(base_dir, self.feature_path)
        print("Feature extraction complete")
        self.feature_ready.set()

    def run_vslam(self):
        """Run VSLAM processing"""
        print("VSLAM thread waiting for features...")
        self.feature_ready.wait()
        print("Starting VSLAM processing...")
        try:
            vslam_main()
            self.completion_queue.put(('VSLAM', True, None))
        except Exception as e:
            self.completion_queue.put(('VSLAM', False, str(e)))
        print("VSLAM processing complete")

    def run_classifier(self):
        """Run object classification"""
        print("Object classifier thread waiting for features...")
        self.feature_ready.wait()
        print("Starting object classification...")
        try:
            classifier_main()
            self.completion_queue.put(('Classifier', True, None))
        except Exception as e:
            self.completion_queue.put(('Classifier', False, str(e)))
        print("Object classification complete")

    def run_parallel_processing(self):
        """Main method to run all processes"""
        # Create threads
        feature_thread = threading.Thread(target=self.extract_features)
        vslam_thread = threading.Thread(target=self.run_vslam)
        classifier_thread = threading.Thread(target=self.run_classifier)

        # Start all threads
        start_time = time.time()
        feature_thread.start()
        vslam_thread.start()
        classifier_thread.start()

        # Wait for processing threads to complete
        feature_thread.join()
        vslam_thread.join()
        classifier_thread.join()

        # Check for any errors
        while not self.completion_queue.empty():
            process, success, error = self.completion_queue.get()
            if not success:
                print(f"Error in {process}: {error}")
            else:
                print(f"{process} completed successfully")

        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

def main():
    # Enable CUDA if available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        # Set CUDA device for all threads
        torch.set_num_threads(1)  # Prevent thread oversubscription
    else:
        print("CUDA is not available. Using CPU.")

    processor = FeatureProcessor()
    processor.run_parallel_processing()

if __name__ == "__main__":
    main()