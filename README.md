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
- **Size**: 600 image pairs
- **Format**: 
  - IR: 24-bit grayscale
  - Visible: 24-bit RGB
- **Resolution**: 512×512 pixels
- **Alignment**: All pairs are precisely registered
- **Annotations**: 8,891 labeled instances across 7 categories
  > Note: We welcome feedback on annotation quality

### Statistics

| Category | Objects | Images | Coverage |
|:---------|--------:|-------:|----------:|
| Car | 8,046 | 419 | 86.75% |
| Person | 526 | 118 | 24.43% |
| Bus | 113 | 79 | 16.36% |
| Truck | 157 | 86 | 17.81% |
| E-Bike | 4 | 2 | 0.41% |
| Tent | 138 | 54 | 11.18% |
| Boat | 7 | 7 | 1.45% |
| **Total** | **8,991** | **483** | - |

## 📥 Download

<div align="center">

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/10N7X3N-aM3qbwXHOkUXckia9EvL1Kjq3?usp=sharing)
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
