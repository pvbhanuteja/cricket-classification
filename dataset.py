import os, torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path='./cricket_data.pt', label2id=None, id2label=None, type='main'):
        self.data_list = torch.load(data_path)
        
        # If label2id and id2label are not provided, create default mappings.
        if type == 'main':
            for item in self.data_list:
                item['label'] = item['label'].split(" ")[0]
        else:
            for item in self.data_list:
                item['label'] = item['label'].split(" ")[1]
        labels = sorted(set([item['label'] for item in self.data_list]))
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}    
        self.label2id = label2id
        self.id2label = id2label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        waveform_array = self.data_list[idx]['array']
        label = self.data_list[idx]['label']
        label_id = self.label2id[label]  # Convert label to id
        return torch.tensor(waveform_array, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)
