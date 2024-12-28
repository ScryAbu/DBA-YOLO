# YOLOv7 Fire Detection Model (DBA-YOLO)

This repository contains the implementation of the DBA-YOLO algorithm, a lightweight fire detection model based on YOLOv7-Tiny, enhanced with structural re-parameterization and multiple attention mechanisms. DBA-YOLO is designed for real-time fire and smoke detection in complex environments, achieving high precision and efficiency.

## Overview of DBA-Fire Dataset

The DBA-Fire dataset is a comprehensive collection designed for fire and smoke detection in real-world scenarios. It consists of 3905 high-quality images with YOLO-format annotations. The dataset covers a wide variety of fire-related environments, including indoor and outdoor fires, ranging from small flames to large-scale fires in complex settings such as city roads, buildings, supermarkets, and parking lots.

### Dataset Highlights
- **Size**: 3905 images
- **Categories**:
  - Flames
  - Smoke
  - Both flames and smoke
- **Scenes**: Diverse fire scenarios in both indoor and outdoor settings with varying complexities.
- **Data Format**:
  - **Images**: JPEG format (stored in `DBA-Fire/JPEGImages/`)
  - **Labels**: YOLO-format labels (stored in `DBA-Fire/txt/`), one label file per image.

### Data Processing Pipeline
1. **Data Collection**: Images and videos of fire-related scenes were gathered using a web crawler, and video frames were extracted at a rate of 25 FPS.
2. **Frame Segmentation**: Frames were sampled every 3 frames to ensure variability.
3. **Data Cleansing**: Images were filtered based on several criteria such as object coverage and noise ratio.
4. **Object Labeling**: Images were manually annotated for flames and smoke using LabelImg, and converted to YOLO-format labels.

### Intended Use
This dataset is primarily designed for training, fine-tuning, and evaluating fire and smoke detection models.

The DBA-Fire dataset will be available on Huggingface and GitHub after the acceptance of the associated paper. For more details, please visit [DBA-YOLO Dataset](https://github.com/ScryAbu/DBA-YOLO-Dataset).

---

## DBA-YOLO Model Overview

DBA-YOLO is an enhanced version of YOLOv7-Tiny, incorporating attention mechanisms and structural re-parameterization for improved flame detection. The model introduces a new feature extraction module to boost performance and a focal modulation network to increase the receptive field of the detection layer. 

**Key Highlights**:
- Average detection precision: 85.4%
- Mean Average Precision (mAP50): 85.6%
- Faster than YOLOv7 by 27% with an average detection time of 6.5ms per frame.
- Detection of fire, fire-like, and smoke-like objects with high precision and minimized false positives.

### Abstract
In response to the challenges of detecting smoke-like objects in fire scenes and fire detection under various natural lighting conditions, we propose an improved fire detection algorithm based on YOLOv7-Tiny, named DBA-YOLO. This novel approach introduces an attention feature extraction module that combines structural re-parameterization, extending the feature extraction network architecture. It enhances feature propagation for flame target recognition and improves the performance of the network model. The integration of a focal modulation network into the head detection layer facilitates better feature exchange and enlarges the receptive field of the detection layer. By replacing the original model's detection head with a variable attention detection head, we significantly increase detection precision.

---

## Training the Model

To train the DBA-YOLO model, follow the instructions below. The following configuration parameters are used for training:
- **Epochs**: 300
- **Batch Size**: 16
- **Image Size**: 640x640
- **Optimizer**: Adam
- **Workers**: 6

You can adjust the model path in `train.py` at line 529 to point to the correct location of your weights.

### Training Command
Ensure that your environment is set up properly and dependencies are installed. Once ready, run the following command to start training:

```bash
python train.py --epochs 300 --img 640 --batch 16 --optimizer adam --workers 6 --data /path/to/dataset.yaml --weights /path/to/initial/weights.pt
