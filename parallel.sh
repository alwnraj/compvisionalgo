#!/bin/bash

# Create/clear the total_runtime.txt file
echo "" > parallel_runtime1.txt

#echo "Running pllqueue.py..."
echo "Running parallelstream.py..."
start_time=$(date +%s.%N)
python parallelstream.py
end_time=$(date +%s.%N)

runtime1=$(echo "$end_time - $start_time" | bc)
echo "Runtime for feature extraction and object classification & vslam: $runtime1 seconds" >> parallel_runtime1.txt