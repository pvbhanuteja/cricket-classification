import os
import torchaudio, torch
from utils import remove_human_voice
# Loading the cricket audio .wav files into waveform list
wav_dir = "/Users/bhanu/repos/ast_classification/data/"

waveform_list = []
num_iter = 0

SR = 16000

cricket_names = []  # use parallel array to keep track of cricket name for each audio file
data = []
for wav_file in sorted(os.listdir(wav_dir)):
    num_iter += 1
    label_name = ''
    print("ITERATION NUMBER: ", str(num_iter), " FILE NAME: ", wav_file)
    if 'mp3' in wav_file:
        continue

    # Xenogryllus MCL files (no species name) assumed to be Xenogryllus uniparitus species
    if wav_file.find("Xenogryllus") != -1 and wav_file.find("(MCL)") != -1:
        cricket_names.append("Xenogryllus" + " " + "unipartitus" + " " + "MCL")
        label_name = "Xenogryllus" + " " + "unipartitus" + " " + "MCL"
    else:
        split_file = wav_file.split(" ")
        if wav_file.find("MCL") != -1:  # if file is from MCL
            cricket_names.append(
                split_file[2] + " " + split_file[3] + " " + "MCL")
            label_name = split_file[2] + " " + split_file[3] + " " + "MCL"
        else:  # if file is from SINA
            cricket_names.append(
                split_file[2] + " " + split_file[3] + " " + "SINA")
            label_name = split_file[2] + " " + split_file[3] + " " + "SINA"

    waveform, sample_rate = torchaudio.load(wav_dir + wav_file)
    wav = remove_human_voice(waveform, sample_rate)
    print(wav.shape)
    #trim 1 second from beginning and end of audio file
    wav = wav[SR:-SR]
    # convert above wav into a chunks of 15 seconds each with 5 seconds overlap
    for i in range(0, len(wav), 5 * SR):
        if i + 15 * SR < len(wav):
            waveform = wav[i: i + 15 * SR]
            data.append({"array": waveform.squeeze().numpy(), "label": label_name})
        else:
            waveform = wav[i: len(wav)]
            if len(waveform) < 10 * SR:
                continue
            else:
                data.append({"array": waveform.squeeze().numpy(), "label": label_name})
                
# save the data list as a .pt file
torch.save(data, "./cricket_data.pt")
