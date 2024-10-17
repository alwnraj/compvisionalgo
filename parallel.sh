#!/bin/bash

# Create/clear the total_runtime.txt file
echo "" > parallel_runtime.txt

echo "Running feature_extract.py and parallel.py..."
start_time=$(date +%s.%N)
python feature_extract.py
python parallel.py
end_time=$(date +%s.%N)

runtime1=$(echo "$end_time - $start_time" | bc)
echo "Runtime for feature extraction and object classification & vslam: $runtime1 seconds" >> parallel_runtime.txt