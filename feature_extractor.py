import torch
import numpy as np
from transformers import AutoFeatureExtractor
import argparse

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

def process_samples_in_batches(samples, batch_size):
    all_processed_samples = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
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

def main(input_dir, output_dir):
    samples = torch.load(input_dir)
    all_processed_samples = process_samples_in_batches(samples, 16)
    print("len of all_processed_samples: ", len(all_processed_samples))
    print("shape of all_processed_samples: ", all_processed_samples[0]['array'].shape, all_processed_samples[0]['label'])
    torch.save(all_processed_samples, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing the input data")
    parser.add_argument("output_dir", help="Directory to save the processed data")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
