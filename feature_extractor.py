import torch
import numpy as np
from transformers import AutoFeatureExtractor
import argparse
from tqdm import tqdm
import json

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

def process_samples_in_batches(samples, batch_size):
    all_processed_samples = []
    num_samples = len(samples)
    num_batches = int(np.ceil(num_samples / float(batch_size)))
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start = i * batch_size
        end = min(start + batch_size, num_samples)
        batch_samples = samples[start:end]
        
        arrays = [sample['array'] for sample in batch_samples]
        inputs = feature_extractor(
            arrays, sampling_rate=16000, return_tensors="np"
        )
        for input_values, sample in zip(inputs['input_values'], batch_samples):
            processed_sample = {
                'array': input_values,
                'label': sample['label']
            }
            all_processed_samples.append(processed_sample)
    return all_processed_samples

def main(input_dir, output_dir, batch_size=16):
    samples = torch.load(input_dir)
    all_processed_samples = process_samples_in_batches(samples, batch_size)
    print("len of all_processed_samples: ", len(all_processed_samples))
    print("shape of all_processed_samples: ", all_processed_samples[0]['array'].shape, all_processed_samples[0]['label'])
    torch.save(all_processed_samples, output_dir + 'cricket_data_feature_extracted.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing the input data")
    parser.add_argument("output_dir", help="Directory to save the processed data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.batch_size)
    
    # Load the configuration file
    with open('config.json') as config_file:
        config = json.load(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['train', 'test'], required=True, help='Specify whether to process train or test data')
    parser.add_argument('--data_path', default=None, help='Path to the input data directory')
    parser.add_argument('--output_path', default=None, help='Path to the output directory')
    parser.add_argument('--batch_size', type=int, default=config['feature_extractor']['batch_size'], help='Batch size for processing')
    args = parser.parse_args()

    data_type = args.data_type
    batch_size = args.batch_size
    data_path = args.data_path
    output_path = args.output_path

    if args.data_path is None:

        data_path = config['feature_extractor']['train_data_pt_path'] if data_type == 'train' else config['feature_extractor']['test_data_pt_path']

    if args.output_path is None:
        
        output_path = config['feature_extractor']['train_features_output_path'] if data_type == 'train' else config['feature_extractor']['test_features_output_path']

    main(data_path, output_path, batch_size)
