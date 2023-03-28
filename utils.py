import torch
import torchaudio
import random

USE_ONNX = False
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks,
 ) = utils

def split_data(data, train_pct=0.75, test_pct=0.20, val_pct=0.05,seed=42):
    random.seed(seed)
    if train_pct + test_pct + val_pct != 1.0:
        raise ValueError("The sum of percentages should be equal to 1.")

    # Shuffle the data
    random.shuffle(data)

    # Calculate the number of samples for train, test, and validation sets
    total_samples = len(data)
    train_size = int(total_samples * train_pct)
    test_size = int(total_samples * test_pct)
    val_size = total_samples - train_size - test_size

    # Split the data into train, test, and validation sets
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]

    return train_data, test_data, val_data

def drop_chunks(tss,wav):
    if len(tss) == 0:
        return wav
    chunks = []
    cur_start = 0
    for i in tss:
        chunks.append((wav[cur_start: i['start']]))
        cur_start = i['end']
    return torch.cat(chunks)

def remove_human_voice(wav, sr):
    # # Handle multi-channel audio
    print(wav.shape)
    wav = wav.mean(dim=0) if wav.ndim > 1 else wav
    if sr != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resample_transform(wav)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    print(speech_timestamps)
    final_wav = drop_chunks(speech_timestamps, wav)
    return final_wav.unsqueeze(0)

if __name__== "__main__":
    wav, sr = torchaudio.load("/Users/bhanu/repos/ast_classification/data/Copy of Allonemobius allardi 3 (MCL).WAV")
    print(sr)
    wav1 = remove_human_voice(wav, sr)
    print(wav1.shape)
    torchaudio.save("./test/audio_clean_1.wav", wav1, 16000)