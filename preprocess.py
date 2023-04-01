import os
import torchaudio
import torch
from utils import remove_human_voice
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

wav_dir = "./data/"

SR = 16000
SECONDS_TO_TRIM = 10
SECONDS_TO_OVERLAP = 5

def process_wav_file(wav_file):
    data = []
    label_name = ''
    
    if wav_file.find("Xenogryllus") != -1 and wav_file.find("(MCL)") != -1:
        label_name = "Xenogryllus" + " " + "unipartitus" + " " + "MCL"
    else:
        split_file = wav_file.split(" ")
        if wav_file.find("MCL") != -1:
            label_name = split_file[2] + " " + split_file[3] + " " + "MCL"
        else:
            label_name = split_file[2] + " " + split_file[3] + " " + "SINA"

    waveform, sample_rate = torchaudio.load(wav_dir + wav_file)
    wav = remove_human_voice(waveform, sample_rate).squeeze(0)
    wav = wav[SR:-SR]

    for i in range(0, len(wav), SECONDS_TO_OVERLAP * SR):
        if i + SECONDS_TO_TRIM * SR < len(wav):
            waveform = wav[i: i + SECONDS_TO_TRIM * SR]
            data.append({"array": waveform.squeeze().numpy(), "label": label_name})
        else:
            waveform = wav[i: len(wav)]
            if len(waveform) < 10 * SR:
                continue
            else:
                data.append({"array": waveform.squeeze().numpy(), "label": label_name})

    return data

def preprocess(wav_dir,n_jobs=cpu_count()):
    wav_files = sorted(os.listdir(wav_dir))
    data = []

    with Pool(n_jobs) as pool:
        results = pool.imap_unordered(process_wav_file, wav_files)

        for result in tqdm(results, total=len(wav_files), desc="Processing wav files"):
            data.extend(result)

        pool.close()
        pool.join()

    torch.save(data, "./cricket_data.pt")
if __name__ == "__main__":
    preprocess(wav_dir,n_jobs=1)
    print("done")