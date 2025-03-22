# FS-Diff: Semantic Guidance and Clarity-Aware Simultaneous Multimodal Image Fusion and Super-Resolution

<div align="center">
  <h2>AVMS Dataset</h2>
</div>

<div align="center">
  <img src="assets/1.png" alt="AVMS" width="80%">
</div>

## 📊 Preview

<div align="center">
  <img src="assets/2.png" alt="preview" width="90%">
</div>

## 📋 Details

### 📷 Sensor
DJI M30T, Its infrared camera operates within a spectral range of 8-14µm. The optical centers of the infrared and visible cameras are 3 cm apart.

### 📍 Main Scene
- The aerial photography location is in Foshan, Guangdong, China.

### 📁 Dataset Information
- **Total number of image pairs**: **600** (for fusion, detection, segmentation)
- **Format of images**: 
  - [Infrared] 24-bit grayscale bitmap
  - [Visible] 24-bit color bitmap
- **Image size**: **512 × 512** pixels
- **Registration**: **All image pairs are registered.** 

## 📥 Download

<div align="center">

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)]()
[![Baidu Yun](https://img.shields.io/badge/Baidu%20Yun-2932E1?style=for-the-badge&logo=baidu&logoColor=white)]()

</div>

## 🚀 Usage

### Environment Setup
```python
pip install -r requirement.txt
```

### infer
```python
# run the script
python infer.py -c [config file]
```
### train
```python
# run the script
python sr.py -p train -c [config file] -enable_wandb -log_eval
```
