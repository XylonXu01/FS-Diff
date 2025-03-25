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
- **Labeling**: **8891 segmentation and detection labels** have been manually labeled, containing 7 kinds of targets: **{car, person, bus,
  truck, electric_bicycle, tent, boat}**. (Limited by manpower, some targets may be mismarked or missed. We would appreciate if you
  would point out wrong or missing labels to help us improve the dataset)

| category | target quantity | file quantity | proportion |
| --- | --- | --- | --- |
| car | 8046 | 419 | 86.75% |
| person | 526 | 118 | 24.43% |
| bus | 113 | 79 | 16.36% |
| truck | 157 | 86 | 17.81% |
| electric_bicycle | 4 | 2 | 0.41% |
| tent | 138 | 54 | 11.18% |
| boat | 7 | 7 | 1.45% |
| **total** | 8991 | 483 | - |


## 📥 Download

<div align="center">

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)]()
[![Baidu Yun](https://img.shields.io/badge/Baidu%20Yun-2932E1?style=for-the-badge&logo=baidu&logoColor=white)](https://pan.baidu.com/s/1k2x70FSdWtKpwwh8Kh0GVA?pwd=idrs)

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
