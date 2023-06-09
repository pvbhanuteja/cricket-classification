#!/bin/bash

# Alert the user to activate their virtual environment
echo "Please ensure your virtual environment is activated before running this script."

# Print and run the first command
echo "Running command: python preprocess.py --data_type train"
python preprocess.py --data_type train

# Print and run the second command
echo "Running command: python preprocess.py --data_type test"
python preprocess.py --data_type test

# Print and run the third command
echo "Running command: python feature_extractor.py --data_type train"
python feature_extractor.py --data_type train

# Print and run the fourth command
echo "Running command: python feature_extractor.py --data_type test"
python feature_extractor.py --data_type test

# Print and run the fifth command
echo "Running command: python main.py"
python main.py

# Print completion message
echo "All commands have completed successfully!"
