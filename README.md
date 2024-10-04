# compvisionalgo
Computer Vision algorithm implementation with rgb-d dataset

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/alwnraj/compvisionalgo.git
```

### Step 2: Setting up a virtual environment
```bash
python3 -m venv slam_env
source slam_env/bin/activate
```

### Step 3: Installing the required libraries
```bash
pip install -r requirements.txt
```

### Step 4: Running the program
```bash
python3 deepslam.py
```

- OR you could also run the Efficientnetalgo

```bash
python3 efficientalgo.py
```

### simple_efficientnet

- Removed Feature Extraction: The EfficientNetFeatureExtractor class and any references to feature extraction have been removed.

- Direct Pose Prediction: The EfficientNetRGBOnly class now directly predicts the pose (x, y, z) using the EfficientNet model's classifier, without any intermediate feature extraction.

- Simplified SLAM: The SLAM class has been renamed to SimpleSLAM to reflect the removal of the mapping and feature-based aspects. The update method now directly takes RGB images and ground truth poses.