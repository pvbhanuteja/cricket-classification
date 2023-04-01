from transformers import AutoFeatureExtractor
import torch
import numpy as np

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

samples = torch.load("./cricket_data.pt")
print("len of samples: ", len(samples))
print("shape of samples: ", samples[0]['array'].shape, samples[0]['label'], samples[0]['array'].dtype)

# Function to process samples in batches
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

# Process 16 batches at once
batch_size = 16
all_processed_samples = process_samples_in_batches(samples, batch_size)
print("len of all_processed_samples: ", len(all_processed_samples))
print("shape of all_processed_samples: ", all_processed_samples[0]['array'].shape, all_processed_samples[0]['label'])
torch.save(all_processed_samples, "./cricket_data_feature_extracted.pt")
