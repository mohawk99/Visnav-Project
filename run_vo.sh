#!/bin/zsh

# List of dataset paths
dataset_paths=("data/V1_01_easy/mav0/" "data/V1_02_medium/mav0/" "data/V2_01_easy/mav0/" "data/MH_01_easy/" "data/MH_02_easy/mav0" "data/MH_03_medium/mav0")

# List of corresponding sensor values
sensor_values=("vicon0" "vicon0" "vicon0" "leica0" "leica0" "leica0")

# Ensure both lists have the same length
if [ ${#dataset_paths[@]} -ne ${#sensor_values[@]} ]; then
  echo "Error: The number of dataset paths and sensor values must be the same."
  exit 1
fi

reset && cd build && make -j2 && cd ..

# Loop through each dataset path and corresponding sensor value
for ((i=1; i<${#dataset_paths[@]}; i++)); do
    dataset_path="${dataset_paths[$i]}"
    sensor_value="${sensor_values[$i]}"
    echo "Running command with dataset-path: $dataset_path and sensor: $sensor_value"

    # Execute the command and capture its return code
    ./build/odometry --dataset-path "$dataset_path" --show-gui false --save-loop-pairs false --sensor "$sensor_value"
    return_code=$?

    # Check if the command returned an error (non-zero return code)
    if [ $return_code -ne 0 ]; then
        echo "Error: The command failed with return code $return_code"
        # You can choose to exit the script here or take other actions based on the error
        # exit 1
    fi
done