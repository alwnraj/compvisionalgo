import threading
import queue
import torch
import time
import logging
from feature_extract_2 import extract_features_and_yield
from vslam_stream import StreamingVSLAM
from object_classifier_stream import StreamingObjectClassifier
from vslam import EfficientNetVSLAM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class FeatureQueue:
    """Manages the flow of processed image features between different parts of the system."""
    def __init__(self, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
        self.finished = threading.Event()

    def put(self, item):
        if item is not None:
            self.queue.put(item)
        else:
            self.mark_finished()

    def get(self):
        return self.queue.get()

    def task_done(self):
        self.queue.task_done()

    def join(self):
        self.queue.join()

    def mark_finished(self):
        self.finished.set()

    def is_finished(self):
        return self.finished.is_set()

class FeatureExtractorProducer(threading.Thread):
    """Processes images one by one and puts their features onto the queue."""
    def __init__(self, dataset_path, feature_queue, device='cpu'):
        super().__init__()
        self.dataset_path = dataset_path
        self.device = device
        self.feature_queue = feature_queue

    def run(self):
        try:
            logging.info("Feature extraction started.")
            for idx, features in enumerate(extract_features_and_yield(self.dataset_path, device=self.device)):
                self.feature_queue.put(features)
                logging.info(f"Processed features for image {idx}.")
        except Exception as e:
            logging.error(f"Problem during feature extraction: {str(e)}")
            raise
        finally:
            self.feature_queue.put(None)
            logging.info("Finished extracting features.")

class VSLAMConsumer(threading.Thread):
    """Processes each set of features for VSLAM."""
    def __init__(self, feature_queue, device):
        super().__init__()
        vslam_model = EfficientNetVSLAM().to(device)
        self.feature_queue = feature_queue
        self.vslam = StreamingVSLAM(model=vslam_model, device=device)

    def run(self):
        logging.info("VSLAM processing started.")
        try:
            while not self.feature_queue.is_finished():
                features = self.feature_queue.get()
                if features is None:
                    break
                results = self.vslam.process(features)
                logging.info(f"VSLAM results: {results}")
                self.feature_queue.task_done()
        except Exception as e:
            logging.error(f"VSLAM error: {str(e)}")
            raise
        finally:
            logging.info("VSLAM processing completed.")

class ClassifierConsumer(threading.Thread):
    """Processes each set of features for object classification."""
    def __init__(self, feature_queue, device):
        super().__init__()
        self.feature_queue = feature_queue
        self.classifier = StreamingObjectClassifier(device=device)

    def run(self):
        logging.info("Object classification started.")
        feature_idx = 0
        try:
            while not self.feature_queue.is_finished():
                features = self.feature_queue.get()
                if features is None:
                    break
                self.classifier.classify_feature(features, feature_idx)
                self.feature_queue.task_done()
                feature_idx += 1
        except Exception as e:
            logging.error(f"Classifier error: {str(e)}")
            raise
        finally:
            logging.info("Object classification completed.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    dataset_path = 'rgbd_dataset_freiburg1_xyz'
    feature_queue = FeatureQueue(maxsize=10)

    extractor = FeatureExtractorProducer(dataset_path, feature_queue, device=device)
    vslam_consumer = VSLAMConsumer(feature_queue, device)
    classifier_consumer = ClassifierConsumer(feature_queue, device)

    start_time = time.time()

    try:
        extractor.start()
        vslam_consumer.start()
        classifier_consumer.start()

        extractor.join()
        vslam_consumer.join()
        classifier_consumer.join()
        feature_queue.join()

    except KeyboardInterrupt:
        logging.info("\nReceived shutdown signal. Stopping all processes...")
    finally:
        end_time = time.time()
        logging.info(f"\nTotal time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()