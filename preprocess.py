import os
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint
from tqdm import tqdm
import time
import argparse


SR = 16000
SECONDS_TO_TRIM = 10
SECONDS_TO_OVERLAP = 5
NUM_PROCESS = 40  # set to the number of CPU cores in the machine

torch.set_num_threads(1)

def drop_chunks(tss, wav):
    if len(tss) == 0:
        return wav
    chunks = []
    cur_start = 0
    for i in tss:
        chunks.append((wav[cur_start: i['start']]))
        cur_start = i['end']
    # Append the remaining part of the wav tensor after the last segment
    chunks.append(wav[cur_start:])
    return torch.cat(chunks)


def get_model_and_utils():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                   model='silero_vad',
                                   force_reload=False,
                                   onnx=False,
                                   verbose=False)
    return model, utils


def process_wav_file(wav_file: str):
    # print("Processing file: " + wav_file)
    model, utils = get_model_and_utils()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    label_name = determine_label_name(wav_file)
    wav = load_wav_file(wav_file)
    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SR)
    final_wav = drop_chunks(speech_timestamps, wav)
    if(len(final_wav) < 1 * SR):
        print("Getting no data from : ",wav_file)
    wav = final_wav[SR:-SR]

    return extract_data_chunks(wav, label_name)


def determine_label_name(wav_file):
    if wav_file.find("Xenogryllus") != -1 and wav_file.find("(MCL)") != -1:
        return "Xenogryllus" + " " + "unipartitus" + " " + "MCL"
    else:
        split_file = wav_file.split(" ")
        if wav_file.find("MCL") != -1:
            return split_file[2] + " " + split_file[3] + " " + "MCL"
        else:
            return split_file[2] + " " + split_file[3] + " " + "SINA"


def load_wav_file(wav_file):
    wav, sample_rate = torchaudio.load(wav_dir + wav_file)
    wav = wav.mean(dim=0) if wav.ndim > 1 else wav
    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        wav = resample_transform(wav)
    return wav


def extract_data_chunks(wav, label_name):
    data = []
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


def main(input_dir, save_dir):
    global wav_dir
    wav_dir = input_dir
    futures = []
    data = []

    with ProcessPoolExecutor(max_workers=NUM_PROCESS) as ex:
        wav_files = sorted(os.listdir(wav_dir))
        for wav_file in wav_files:
            futures.append(ex.submit(process_wav_file, wav_file))

        for finished in tqdm(as_completed(futures), total=len(futures), desc="Processing WAV files"):
            result = finished.result()
            data.extend(result)

    # pprint(data)
    pprint(len(data))
    torch.save(data, save_dir + "data.pt")


# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing the WAV files")
    parser.add_argument("output_dir", help="Directory to save the vad processed data")
    args = parser.parse_args()
    start_time = time.time()
    main(args.input_dir, args.output_dir)
    end_time = time.time()
    pprint(f"Execution time: {end_time - start_time:.4f} seconds")
