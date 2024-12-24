# VanillaNet: Accurate AOD Estimation

## 🌟 Project Overview
This project was developed as part of a hackathon to improve **Aerosol Optical Depth (AOD)** estimation using Sentinel-2 satellite imagery and the AERONET dataset. AOD plays a critical role in understanding atmospheric conditions, climate change, and public health. Our contribution, **VanillaNet**, achieved a rank of **45th place**, demonstrating significant innovation and performance.

By leveraging deep learning, specifically a novel neural network architecture, this project aims to:
- Enhance the accuracy of AOD predictions.
- Contribute to environmental monitoring and decision-making.
- Advance methodologies for AOD estimation, aiding global efforts in climate research and environmental protection.

---

## 🚀 Key Features
- **Custom Neural Network**: Developed **VanillaNet**, a state-of-the-art architecture tailored for AOD estimation.
- **Sentinel-2 and AERONET Data Integration**: Utilized satellite imagery and ground-truth AERONET data for robust training and validation.
- **Scalable and Deployable**: The model is designed for deployment in real-world environmental monitoring systems.

---

## 📂 Dataset Details
### Sentinel-2 Images
- Contains 13 bands: `'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'`.
- Images processed to mask clouds from 2016/01/01 to 2024/05/01.
- For more details, visit the official Sentinel-2 documentation [here]([https://sentinel.esa.int/](https://solafune.com/competitions/ca6ee401-eba9-4f7d-95e6-d1b378a17200?menu=data&tab=)).


## 📊 Model Architecture
**VanillaNet** was specifically designed for this competition, featuring:
- **Custom Activation Layers**: Leveraging series-informed activation functions for better feature extraction.
- **Deployability**: Designed for efficient inference using batch normalization fusion.
- **Adaptive Pooling**: Handles varying image resolutions.

### Core Components
1. **Custom Activation Function**: Enhances expressiveness with learnable convolution-based activations.
2. **Deep Feature Extraction**: Employs multiple stages of convolutional layers with progressive down-sampling.
3. **Leaky ReLU Optimization**: Ensures stability and faster convergence during training.

---


## 📈 Results
- **Final Placement**: 45th place.

---

## 🛠️ Setup and Usage
### Prerequisites
- Python >= 3.8
- PyTorch >= 1.11.0
- Other dependencies in `requirements.txt`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/vanillanet.git
   cd vanillanet
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


