# filename: pllqueue.py

import threading
import queue
import torch
import time
from feature_extract_2 import extract_features_and_yield
from vslam_stream import StreamingVSLAM
from object_classifier_stream import StreamingObjectClassifier
from vslam import EfficientNetVSLAM

class FeatureQueue:
    """
    queue that manages the flow of processed image features between different
    parts of the system.
    """
    def __init__(self, maxsize=10):
        """Sets up the queue with safety features and error handling."""
        self.queue = queue.Queue(maxsize=maxsize)  # The actual queue
        self.finished = threading.Event()  # Flag to signal when all work is done
        self.error = None  # Stores any errors that occur
        self._lock = threading.Lock()  # Ensures thread safety
        self.batch_counter = 0  # Counts how many batches have been processed

    def put(self, item):
        """Adds a new batch of processed features to the queue."""
        if item is not None:
            self.batch_counter += 1
        self.queue.put((self.batch_counter - 1, item))

    def get(self, timeout=None):
        """Retrieves the next batch of features from the queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            raise queue.Empty()

    def task_done(self):
        """Signals that a queued task is complete."""
        self.queue.task_done()

    def mark_finished(self):
        """Marks the queue as finished, signaling no more items will be added."""
        self.finished.set()

    def is_finished(self):
        """Checks whether feature extraction has finished."""
        return self.finished.is_set()

    def set_error(self, error):
        """Sets an error to be handled by other threads."""
        with self._lock:
            self.error = error

    def get_error(self):
        """Retrieves any errors that occurred."""
        with self._lock:
            return self.error


class FeatureExtractorProducer:
    """
- feature_extract_2.py itself is not treated as a thread in pllqueue.py. 
Instead, the feature extraction process from feature_extract_2.py is integrated 
into pllqueue.py and run in a separate thread.

- the feature extraction logic from feature_extract_2.py is executed inside a thread, 
but the file itself is not treated as a thread. It’s just providing the logic that 
is run within the thread created in pllqueue.py.
    """
    def __init__(self, dataset_path, feature_queue, batch_size=32, device='cpu'):
        self.dataset_path = dataset_path  # Where to find the images
        self.batch_size = batch_size  # How many images to process at once
        self.device = device  # Whether to use CPU or GPU
        self.feature_queue = feature_queue  # The queue to put processed features into

    def extract_features(self):
        """Processes images and puts their features onto the queue for other processes."""
        try:
            print("Feature extraction started.")  # Line added to indicate start
            for features in extract_features_and_yield(
                self.dataset_path, 
                batch_size=self.batch_size, 
                device=self.device
            ):
                self.feature_queue.put(features)
                print(f"Processed new batch of images: {features.shape}")
        except Exception as e:
            self.feature_queue.set_error(f"Problem during feature extraction: {str(e)}")
            raise
        finally:
            self.feature_queue.mark_finished()
            self.feature_queue.put(None)  # Signal that we're done
            print("Finished extracting all features from images.")

class VSLAMConsumer(threading.Thread):
    """
    One of two parallel workers that process the extracted features.
    This one handles VSLAM (Visual Simultaneous Localization and Mapping)
    """
    def __init__(self, feature_queue, device):
        super().__init__()
        vslam_model = EfficientNetVSLAM().to(device)  # Initialize the VSLAM model
        self.feature_queue = feature_queue
        self.vslam = StreamingVSLAM(model=vslam_model, device=device)
        self.processed_count = 0

    def run(self):
        """Processes batches from the feature queue for VSLAM."""
        print("VSLAM processing started.")
        try:
            while True:
                try:
                    batch_idx, features = self.feature_queue.get(timeout=1.0)
                    if features is None:  # End of stream
                        break
                    
                    # Process the batch of features
                    results = self.vslam.process_batch(features, batch_idx)
                    self.processed_count += 1
                    self.feature_queue.task_done()

                    # Print stats every 10 processed batches
                    if self.processed_count % 10 == 0:
                        stats = self.vslam.get_results()
                        print(f"VSLAM stats: {stats}")

                except queue.Empty:
                    if self.feature_queue.is_finished():
                        break
                    continue

        except Exception as e:
            self.feature_queue.set_error(f"VSLAM error: {str(e)}")
            raise
        finally:
            # Print final VSLAM stats
            final_stats = self.vslam.get_results()
            print(f"VSLAM finished. Final stats: {final_stats}")

class ClassifierConsumer(threading.Thread):
    """
    The second parallel worker that processes the extracted features.
    This one identifies objects in the images, like recognizing cars,
    people, or buildings in the scene.
    """
    def __init__(self, feature_queue, device):
        super().__init__()
        self.feature_queue = feature_queue
        self.classifier = StreamingObjectClassifier(device=device)
        self.processed_count = 0

    def run(self):
        """Processes batches from the feature queue for object classification."""
        print("Object classification started.")
        try:
            while True:
                try:
                    batch_idx, features = self.feature_queue.get(timeout=1.0)
                    if features is None:  # End of stream
                        break
                    
                    # Process the batch of features
                    results = self.classifier.process_batch(features, batch_idx)
                    self.processed_count += 1
                    self.feature_queue.task_done()

                    # Print stats every 10 processed batches
                    if self.processed_count % 10 == 0:
                        stats = self.classifier.get_results()
                        print(f"Classifier stats: {stats}")

                except queue.Empty:
                    if self.feature_queue.is_finished():
                        break
                    continue

        except Exception as e:
            self.feature_queue.set_error(f"Classifier error: {str(e)}")
            raise
        finally:
            # Print final classification stats
            final_stats = self.classifier.get_results()
            print(f"Classification finished. Final stats: {final_stats}")

def main():

    # Check if we can use a GPU, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_path = 'rgbd_dataset_freiburg1_xyz'  # location of input images
    feature_queue = FeatureQueue(maxsize=10)  # Create the shared queue
    
    # Set up our three workers
    extractor = FeatureExtractorProducer(dataset_path, feature_queue, device=device)
    vslam_consumer = VSLAMConsumer(feature_queue, device)
    classifier_consumer = ClassifierConsumer(feature_queue, device)

    start_time = time.time()
    
    try:
        # Start all processes
        feature_thread = threading.Thread(target=extractor.extract_features)
        feature_thread.start()
        vslam_consumer.start()
        classifier_consumer.start()

        # Wait for all workers to finish their jobs
        feature_thread.join()
        vslam_consumer.join()
        classifier_consumer.join()

        # Check if any problems occurred
        if feature_queue.get_error():
            print(f"An error occurred: {feature_queue.get_error()}")

    except KeyboardInterrupt:
        print("\nReceived shutdown signal. Stopping all processes...")
    finally:
        # Print final statistics
        end_time = time.time()
        print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
        print(f"Number of image batches processed: {feature_queue.batch_counter}")

if __name__ == "__main__":
    main()