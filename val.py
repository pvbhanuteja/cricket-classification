import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from dataset import CustomDataset
from preprocess import SR

# Load the dataset
dataset = CustomDataset(data_path='./cricket_data.pt', type='main')
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Load the trained model
model_path = "trained_model.pt"
model = ASTForAudioClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(audio_data):
    inputs = feature_extractor(audio_data, sampling_rate=SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_ids = torch.argmax(logits, dim=-1).item()
        predicted_label = model.config.id2label[predicted_class_ids]
    
    return predicted_label

# Test the model on a sample from the dataset
sample_idx = 0
waveform, label_id = dataset[sample_idx]
true_label = dataset.id2label[label_id.item()]

predicted_label = predict(waveform)

print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")
