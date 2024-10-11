# filename: object_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

class FeaturesDataset(Dataset):
    """Dataset to load pre-extracted features (no labels)."""
    def __init__(self, features_path): 
        super(FeaturesDataset, self).__init__()
        self.features = torch.load(features_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx] 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10 
    feature_dim = 1280

    # 1. Load pre-extracted features
    features_path = 'features.pt'  
    features = torch.load(features_path, weights_only=True).to(device)

    # 2. Create FeaturesDataset and DataLoader
    dataset = FeaturesDataset(features_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # No need to shuffle for inference

    # 3. Create ObjectClassifier instance
    model = ObjectClassifier(num_classes=num_classes, feature_dim=feature_dim).to(device)
    model.eval()

    # 4. Get predictions and print probabilities for each image
    with torch.no_grad():
        for batch_idx, features in enumerate(dataloader):
            features = features.to(device)  # Move features to device
            outputs = model(features)

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Print probabilities for each image in the batch
            for i in range(probabilities.shape[0]):
                print(f"Image {batch_idx * dataloader.batch_size + i}: {probabilities[i]}") 

if __name__ == "__main__":
    main()