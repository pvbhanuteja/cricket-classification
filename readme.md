<div  align="center">

<h1  align="center">


<br>

cricket songs classification

</h1>

<h3  align="center">📍 Cricket songs classification fined tuned on AST transformers</h3>


<p  align="center">

  

<img  src="https://img.shields.io/badge/Pytorch-09A3D5.svg?style=for-the-badge&logo=pytorch&logoColor=white"  alt="pytorch"  />

<img  src="https://img.shields.io/badge/Git-8CAAE6.svg?style=for-the-badge&logo=git&logoColor=white"  alt="git"  />
<img  src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white"  alt="hypothesis"  />

</p>

  

</div>

  

---

## 📚 Table of Contents

- [📚 Table of Contents](#-table-of-contents)
- [📍Overview](#overview)
  - [Results](#results)
- [⚙️ Project Structure](#️-project-structure)
- [💻 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
  - [💻 Installation](#-installation)
  - [🤖 Training Model](#-training-model)
- [🤝 Contributing](#-contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

  

---

  

## 📍Overview

  

The Cricket Classification GitHub project is an audio classification system that utilizes deep learning techniques to identify and categorize cricket species based on their sound recordings. The project leverages the PyTorch Lightning framework and the ASTForAudioClassification model from Hugging Face's Transformers library to build and train the classifier. The code includes data preprocessing, model training, and evaluation, providing a complete end-to-end solution for cricket sound classification tasks.

### Results

| Experiment              | Test Accuracy |
|-------------------------|---------------|
| 5 genus classification  | 97.00%        |
| 8 genus classification  | 94.40%        |
| 10 genus classification  | 89.51%        |

These results are obtained on test data using an 80:20 train:test split. The train and test waveforms are split into 10-second segments with a 5-second overlap.



---

  

<img  src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-github-open.svg"  width="80"  />

  

## ⚙️ Project Structure

  

```bash

.
├── config.json
├── data
│ ├──  final_features
│ ├──  raw_all_data
│ └──  vad_processed
├── dataset.py
├── feature_extractor.py
├── helpers
│ ├──  data.txt
│ ├──  make_data.py
│ └──  make_data_dir.sh
├── main.py
├── preprocess.py
├── readme.md
├── requirements.txt
├── run_pipeline.sh
├── utils.py
└── val.py
```

---

  

<img  src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-src-open.svg"  width="80"  />

  

## 💻 Modules



| File                 | Summary                                                                                                                                                                                                                                                                               |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [run_pipeline.sh](./run_pipeline.sh)      | Runs complete pipeline. (preprocess, feature extraction and trains the model)                                                                                                                           |
| [preprocess.py](./preprocess.py)        | This script processes a set of audio files for machine learning purposes, using the Silero Voice Activity Detector (VAD) model to extract relevant speech segments.                                                                                                                   |
| [dataset.py](./dataset.py)           | This script defines a CustomDataset class that inherits from PyTorch's Dataset class, tailored for processing audio data related to cricket sounds.                                                                                                                                    |
| [utils.py](./utils.py)             | This script demonstrates how to remove human voice from an audio file using the Silero Voice Activity Detector (VAD) model.                                                                        |
| [feature_extractor.py](./feature_extractor.py) | This script extracts features from audio samples using a pre-trained feature extractor from the `transformers` library. The `process_samples_in_batches` function processes audio samples in batches, applying the feature extractor to each sample and storing the extracted features along with the sample's label. |
| [main.py](./main.py )              | This script trains a cricket audio classifier using a pre-trained ASTForAudioClassification model from the `transformers` library.                                                                                                                                                   |

<hr/>


  

## 🚀 Getting Started



### 💻 Installation

  

1. Clone the readme-ai repository:

```sh

git clone  https://github.com/pvbhanuteja/cricket-classification

```

  

2. Change to the project directory:

```sh

cd  cricket-classification

```

  

3. Install the dependencies:

```sh

pip install  -r  requirements.txt

```

  

### 🤖 Training Model

  

```sh
# Update config.json with correct paths then run shell script
sh run_pipeline.sh

```

<hr  />


## 🤝 Contributing

Check out [CONTRIBUTING.md](./contributing.md) for best practices and instructions on how to contribute to this project.

---

  

## License


This project is licensed under the `MIT` License. 

---

  

## Acknowledgments


- Professor [Dr. Yoonsuck Choe](https://yschoe.github.io/). 
-  This work was supported in part by the [Texas Virtual Data Library (ViDaL)](https://vidal.tamu.edu/) funded by the Texas A&M University Research Development Fund.


---