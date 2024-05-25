#!/bin/bash

echo "Data preparation started."
# Step 1: Data preparation
/home/team18/miniconda3/bin/python /home/team18/M3/data_preparation.py


# Step 2: Model training
echo "Starting the model update process."

/home/team18/miniconda3/bin/python /home/team18/M3/train.py

echo "Model update process completed."
