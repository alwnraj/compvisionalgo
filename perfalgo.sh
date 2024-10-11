#!/bin/bash

# Create/clear the total_runtime.txt file
echo "" > total_runtime.txt

echo "Running feature_extract.py and object_classifier.py..."
start_time=$(date +%s.%N)
python feature_extract.py
python object_classifier.py
end_time=$(date +%s.%N)

runtime1=$(echo "$end_time - $start_time" | bc)
echo "Runtime for feature extraction and object classification: $runtime1 seconds" >> total_runtime.txt

echo "Running feature_extract.py and vslam.py..."
start_time=$(date +%s.%N)
python feature_extract.py
python vslam.py
end_time=$(date +%s.%N)

runtime2=$(echo "$end_time - $start_time" | bc)
echo "Runtime for feature extraction and VSLAM: $runtime2 seconds" >> total_runtime.txt

echo "Total runtimes saved to total_runtime.txt"