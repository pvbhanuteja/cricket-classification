#!/bin/bash

# Alert the user to activate their virtual environment
echo "Please ensure your virtual environment is activated before running this script."

# Print and run the first command
echo "Running command: python preprocess.py ./data/raw_all_data ./data/train.txt ./data/vad_processed/train/"
python preprocess.py ./data/raw_all_data ./data/train.txt ./data/vad_processed/train/

# Print and run the second command
echo "Running command: python preprocess.py ./data/raw_all_data ./data/test.txt ./data/vad_processed/train/"
python preprocess.py ./data/raw_all_data ./data/test.txt ./data/vad_processed/train/

# Print and run the third command
echo "Running command: python feature_extractor.py ./data/vad_processed/train/data.pt ./data/final_features/train/ --batch_size 64"
python feature_extractor.py ./data/vad_processed/train/data.pt ./data/final_features/train/ --batch_size 64

# Print and run the fourth command
echo "Running command: python feature_extractor.py ./data/vad_processed/test/data.pt ./data/final_features/test/ --batch_size 64"
python feature_extractor.py ./data/vad_processed/test/data.pt ./data/final_features/test/ --batch_size 64

# Print and run the fifth command
echo "Running command: python main.py"
python main.py

# Print completion message
echo "All commands have completed successfully!"
