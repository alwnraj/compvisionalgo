#!/bin/bash

# Start time for both scripts
start_time=$(date +%s)

# Run both scripts in the background
python3 efficientnetalgo.py &
PID1=$!

python3 simple_efficientnet.py &
PID2=$!

# Wait for both scripts to finish
wait $PID1
wait $PID2

# End time
end_time=$(date +%s)

# Calculate and print the total runtime
total_runtime=$((end_time - start_time))
echo "Both scripts finished in $total_runtime seconds" > total_runtime.txt
