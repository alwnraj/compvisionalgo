# filename: object_classifier_stream.py

import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import Queue
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class StreamingObjectClassifier:
    def __init__(self, num_classes=1000, feature_dim=1280, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = ObjectClassifier(num_classes, feature_dim).to(device)
        self.model.eval()  # Set model to evaluation mode for classification
        self.results_queue = Queue()  # Queue to store individual results

    def classify_feature(self, features, feature_idx):
        """Classify a single set of features and return the results."""
        logging.info(f"Input feature size: {features.size()}")
        features = features.to(self.device)
        try:
            with torch.no_grad():
                outputs = self.model(features.unsqueeze(0))
                logging.info(f"Model outputs: {outputs}")
                probabilities = F.softmax(outputs, dim=1)
                logging.info(f"Probabilities: {probabilities}")

                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()

                result = {
                    'index': feature_idx,
                    'predicted_class': pred_class,
                    'confidence': confidence
                }
                self.results_queue.put(result)

                logging.info(f"Processed feature {feature_idx}: Class {pred_class} with confidence {confidence:.2f}")
        except Exception as e:
            logging.error(f"Error during classification: {str(e)}")
            raise

    def get_next_result(self):
        """Get the next classification result from the queue, if available."""
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None

class ObjectClassifier(nn.Module):
    def __init__(self, num_classes=1000, feature_dim=1280):
        super(ObjectClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)

def main():
    """Classify a single feature for testing purposes."""
    classifier = StreamingObjectClassifier()

    # Process individual dummy features
    for i in range(5):
        dummy_feature = torch.randn(1280)
        classifier.classify_feature(dummy_feature, feature_idx=i)

if __name__ == "__main__":
    main()
