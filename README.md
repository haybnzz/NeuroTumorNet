# NeuroTumorNet

A deep learning model for brain tumor classification using MRI images.
![NeuroTumorNet Logo](https://github.com/haybnzz/NeuroTumorNet/blob/main/images/NeuroTumorNet.png?raw=true)

[![TensorFlow - NeuroTumorNet](https://img.shields.io/static/v1?label=TensorFlow&message=NeuroTumorNet&style=for-the-badge&logo=tensorflow&logoSize=auto&labelColor=4B4453&color=FF6F61)](https://github.com/haybnzz/NeuroTumorNet)  [![CC BY-NC 4.0 License](https://img.shields.io/static/v1?label=License&message=CC%20BY-NC%204.0&style=for-the-badge&logo=creative-commons&logoSize=auto&labelColor=4B4453&color=FFD166)](https://github.com/haybnzz/NeuroTumorNet/blob/main/LICENSE)  [![Model Download](https://img.shields.io/static/v1?label=Model&message=Download&style=for-the-badge&logo=huggingface&logoSize=auto&labelColor=4B4453&color=06D6A0)](https://huggingface.co/haydenbanz/NeuroTumorNet/resolve/main/brain_tumor_model.h5?download=true)  [![Live Demo](https://img.shields.io/static/v1?label=Live&message=Demo&style=for-the-badge&logo=streamlit&logoSize=auto&labelColor=4B4453&color=118AB2)](https://huggingface.co/spaces/haydenbanz/NeuroTumorNets)  [![Datasets](https://img.shields.io/static/v1?label=Datasets&message=TumorVision&style=for-the-badge&logo=data:image/png;base64,...&logoSize=auto&labelColor=4B4453&color=EF476F)](https://huggingface.co/datasets/haydenbanz/TumorVisionDatasets/tree/main)  [![GitHub Issues](https://img.shields.io/github/issues/haybnzz/NeuroTumorNet?style=for-the-badge&logo=github&logoSize=auto&labelColor=4B4453&color=073B4C)](https://github.com/haybnzz/NeuroTumorNet/issues)  [![GitHub Stars](https://img.shields.io/github/stars/haybnzz/NeuroTumorNet?style=for-the-badge&logo=github&logoSize=auto&labelColor=4B4453&color=EF476F)](https://github.com/haybnzz/NeuroTumorNet/stargazers) ![Profile Views](https://komarev.com/ghpvc/?username=haybnzz&style=for-the-badge&logo=github&logoSize=auto&labelColor=4B4453&color=FFD166)  [![Website](https://img.shields.io/static/v1?label=Website&message=Hay.Bnz&style=for-the-badge&logo=data:image/png;base64,...&logoSize=auto&labelColor=4B4453&color=EF233C)](https://haybnz.glitch.me/)  [![Model Download - Kaggle](https://img.shields.io/static/v1?label=Kaggle&message=Download&style=for-the-badge&logo=kaggle&logoSize=auto&labelColor=4B4453&color=20BEFF)](https://www.kaggle.com/models/haydenbanz/neurotumornet)  [![Paper](https://img.shields.io/static/v1?label=Paper&message=GitHub&style=for-the-badge&logo=github&logoSize=auto&labelColor=4B4453&color=FFD700)](blob:https://github.com/eaddddc7-df41-49f0-9658-15a716ec46de)

## Description

🧠 Classify brain tumors with **NeuroTumorNet**! 🩻 Powered by a CNN built with TensorFlow 🤖, this tool analyzes MRI scans to detect Glioma, Meningioma, Pituitary, or No Tumor. 🚀 Upload an image via the Streamlit UI 🌐 and get instant predictions with confidence scores! ✨ Download the model or explore the live demo and datasets below. 🖥️


## Overview

NeuroTumorNet is a CNN-based tool that classifies brain MRI images into four categories:
- Glioma tumor
- Meningioma tumor
- No tumor
- Pituitary tumor

The model uses a convolutional neural network architecture built with TensorFlow and Keras to provide accurate tumor classification.

## 🔍 Features

- Automatic detection and classification of brain tumors
- Support for multiple tumor types (glioma, meningioma, pituitary)
- User-friendly web interface for image upload and analysis
- High accuracy brain tumor classification using convolutional neural networks

## 📋 Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [Model](#-model)
- [Dataset](#-dataset)
- [License](#-license)
- [Support](#-support)
- [Contributors](#-contributors-and-developers)

## 🔧 Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/haybnzz/NeuroTumorNet/
cd NeuroTumorNet
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
   - Option 1: Download directly from Hugging Face:
     ```bash
     wget "https://huggingface.co/haydenbanz/NeuroTumorNet/resolve/main/brain_tumor_model.h5?download=true" -O brain_tumor_model.h5
     ```
   - Option 2: Use the provided script to download and prepare the model:
     ```bash
     python data_to_model.py
     ```
 - Option 3: Download directly from Kaggle:

### Dataset (Optional)
   ```bash
     Donload from above Badge section 
     ```
If you want to train the model yourself or test it with the original dataset, you can download the brain tumor MRI dataset from the provided data link in the repository.

## Usage

### Running the Web Application

1. After installation, start the web application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an MRI image through the web interface to get the tumor classification result.


```
NeuroTumorNet/
├── app.py               # Web application for tumor classification
├── data_to_model.py     # Script to download and prepare the model
├── requirements.txt     # Dependencies list
├── brain_tumor_model.h5 # Pre-trained model file
├
└── README.md            # This file
```

## 🧠 Model

NeuroTumorNet uses a deep convolutional neural network architecture designed specifically for medical image classification. The model architecture consists of:

- Multiple convolutional layers with ReLU activation
- Max pooling layers for feature extraction
- Dropout layers to prevent overfitting
- Dense layers for classification

The pre-trained model achieves high accuracy in classifying the four categories of brain MRI images.

## 📊 Dataset

The model was trained on a dataset containing brain MRI images categorized into four classes:
- Glioma tumor
- Meningioma tumor
- Pituitary tumor
- No tumor (normal brain MRI)

To download the dataset for training or testing purposes, visit one of these sources:
- [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

After downloading, place the dataset in a folder named `dataset` with the following structure:

```
dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

# Image Display

Here are the images from the repository:

1. ![Preview Image 1](https://github.com/haybnzz/NeuroTumorNet/raw/refs/heads/main/images/preview_1.webp)
2. ![Preview Image](https://github.com/haybnzz/NeuroTumorNet/raw/refs/heads/main/images/preview.webp)
3. ![Accuracy Image](https://github.com/haybnzz/NeuroTumorNet/raw/refs/heads/main/images/accuracy.webp)


## 📜 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. See the [LICENSE](LICENSE) file for more details.

**Unauthorized use is strictly prohibited.**

📧 Contact: singularat@protn.me

## ☕ Support

Donate via Monero: `45PU6txuLxtFFcVP95qT2xXdg7eZzPsqFfbtZp5HTjLbPquDAugBKNSh1bJ76qmAWNGMBCKk4R1UCYqXxYwYfP2wTggZNhq`

## 👥 Contributors and Developers

[<img src="https://avatars.githubusercontent.com/u/67865621?s=64&v=4" width="64" height="64" alt="haybnzz">](https://github.com/haybnzz)  

[<img src="https://avatars.githubusercontent.com/u/144106684?s=64&v=4" width="64" height="64" alt="Glitchesminds">](https://github.com/Glitchesminds)

## 📝 Citation

If you use NeuroTumorNet in your research, please cite:

```
@software{NeuroTumorNet2025,
  author = {Haybnzz and Glitchesminds},
  title = {NeuroTumorNet: Deep Learning for Brain Tumor Classification},
  url = {https://github.com/haybnzz/NeuroTumorNet},
  year = {2025},
}
```

```
@misc {hay.bnz_2025,
	author       = { {Hay.Bnz} },
	title        = { NeuroTumorNet (Revision 7f9585f) },
	year         = 2025,
	url          = { https://huggingface.co/haydenbanz/NeuroTumorNet },
	doi          = { 10.57967/hf/4899 },
	publisher    = { Hugging Face }
}
```
## Acknowledgments

- Thanks to all contributors to the brain tumor MRI datasets used in training this model
- Built with TensorFlow and Keras
