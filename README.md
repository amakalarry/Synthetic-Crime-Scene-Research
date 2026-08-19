# Generative AI for Crime Scene Simulation: A Synthetic Dataset for Forensic Object Detection

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Dataset: 1,379 Images](https://img.shields.io/badge/Dataset-1%2C379%20Images-blue)](https://github.com/amakalarry/Synthetic-Crime-Scene-Research/tree/main/dataset)
[![Model: YOLOv11s](https://img.shields.io/badge/Model-YOLOv11s-green)](https://github.com/amakalarry/Synthetic-Crime-Scene-Research/tree/main/weights)

---

## Overview

This repository contains the **complete dataset, prompts, training code, and trained model** for a synthetic crime scene object detection system. We use Google Vertex AI **Veo2** (text-to-video) and **Imagen4** (text-to-image) to generate fully synthetic crime scene imagery, then train a **YOLOv11s** detector to identify 13 categories of forensic-relevant objects.

**Key Results:**
- **mAP@0.5:** 0.843
- **mAP@0.5:0.95:** 0.50
- **F1-score:** 0.80 (at confidence threshold 0.56)
- **Best-performing classes:** TV (0.983), Laptop (0.932), Smartwatch (0.931)

This work addresses the critical challenge of **data scarcity in forensic AI** — real crime scene imagery cannot be shared publicly due to privacy, ethical, and legal constraints. Our pipeline demonstrates that fully synthetic data can effectively train forensic object detectors.

---

## Sample Synthetic Images

### Veo2 Text-to-Video Frames
Extracted at 1 FPS from 50 AI-generated 8-second crime scene videos



### Imagen4 Text-to-Image (Class Balancing)
Generated to augment underrepresented classes


### Detection Results (YOLOv11s Inference)
Bounding boxes with confidence scores on held-out test images




---

## Repository Structure

```
Synthetic-Crime-Scene-Research/
│
├── README.md                          # This file
├── LICENSE                            # CC BY-NC 4.0 License
│
├── prompts/
│   ├── all_prompts.csv                # Complete set of 50 Veo2 generation prompts
│   └── imagen4_prompts.csv            # Imagen4 prompts for class balancing
│
├── dataset/
│   ├── images/
│   │   ├── train/                     # 944 training images
│   │   ├── val/                       # 296 validation images
│   │   └── test/                      # 139 test images
│   ├── labels/
│   │   ├── train/                     # YOLO-format annotations (.txt)
│   │   ├── val/
│   │   └── test/
│   └── data.yaml                      # YOLO dataset configuration file
│
├── sample_images/                     # Preview images for quick inspection
│   ├── veo2_sample_01.jpg
│   ├── veo2_sample_02.jpg
│   ├── veo2_sample_03.jpg
│   ├── veo2_sample_04.jpg
│   ├── imagen4_sample_01.jpg
│   ├── imagen4_sample_02.jpg
│   ├── imagen4_sample_03.jpg
│   ├── imagen4_sample_04.jpg
│   ├── detection_sample_01.jpg
│   └── detection_sample_02.jpg
│
├── config/
│   └── train_config.yaml              # YOLOv11s training hyperparameters
│
├── weights/
│   └── best.pt                        # Trained YOLOv11s model weights
│
├── results/
│   ├── results.csv                    # Epoch-by-epoch training metrics
│   ├── confusion_matrix.png           # Confusion matrix
│   ├── BoxPR_curve.png                # Precision-Recall curve
│   ├── BoxF1_curve.png                # F1-Confidence curve
│   └── training_curves.png            # Loss curves
│
└── scripts/
    ├── extract_frames.py              # Frame extraction from Veo2 videos
    ├── train.py                       # Training script
    └── inference.py                   # Run inference on new images
```

---

## Object Classes (13 Categories)

| ID | Class         | Initial (Veo2) | Final (+ Imagen4) | mAP@0.5 |
|:--:|:-------------|:--------------:|:-----------------:|:------:|
| 0  | Body          | 74             | 165               | 0.748  |
| 1  | Laptop        | 229            | 332               | 0.932  |
| 2  | Mobilephone   | 203            | 303               | 0.759  |
| 3  | GlassCup      | 49             | 196               | 0.877  |
| 4  | Keyboard      | 25             | 184               | 0.735  |
| 5  | Camera        | 13             | 134               | 0.850  |
| 6  | Knife         | 45             | 198               | 0.821  |
| 7  | Router        | 19             | 175               | 0.871  |
| 8  | Tablet        | 112            | 140               | 0.837  |
| 9  | Smartwatch    | 160            | 160               | 0.931  |
| 10 | TV            | 19             | 165               | 0.983  |
| 11 | Flashdrive    | 179            | 193               | 0.738  |
| 12 | Smartspeaker  | 138            | 149               | 0.881  |
|    | **Total**     |                |                   | **0.843** |

---

## Quick Start

### Requirements
```bash
pip install ultralytics opencv-python
```

### Run Inference on Your Own Images
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("weights/best.pt")

# Run inference
results = model.predict("path/to/your/image.jpg", conf=0.56)

# Display results
results[0].show()
```

### Train from Scratch
```bash
yolo detect train \
    data=dataset/data.yaml \
    model=yolo11s.pt \
    epochs=300 \
    imgsz=640 \
    batch=8 \
    optimizer=AdamW \
    lr0=0.002 \
    patience=40
```

---

## Dataset Generation Pipeline

```
Text Prompts ──► Veo2 (TTV) ──► 50 Videos ──► Frame Extraction ──► 360 Frames
                                                                        │
                                                                        ▼
                                                              Manual Annotation
                                                              (13 classes, YOLO format)
                                                                        │
                                                                        ▼
                                                              Class Imbalance Detected
                                                              (18:1 ratio)
                                                                        │
Text Prompts ──► Imagen4 (TTI) ──► 949 Additional Images ──────────────►│
                                                                        ▼
                                                              Final Dataset: 1,379 images
                                                              Train: 944 | Val: 296 | Test: 139
                                                                        │
                                                                        ▼
                                                              YOLOv11s Training
                                                              (AdamW, lr=0.002, early stop)
                                                                        │
                                                                        ▼
                                                              mAP@0.5 = 0.843
                                                              F1 = 0.80 @ conf 0.56
```

---

## Training Environment

| Component | Specification |
|:----------|:-------------|
| Platform  | Google Colab Pro |
| GPU       | NVIDIA Tesla T4 / A100 (16–40 GB VRAM) |
| Python    | 3.12 |
| PyTorch   | 2.2 |
| Framework | Ultralytics YOLOv11 |
| Best Model | Epoch 166 / 206 (early stopped) |

---

## Ethical Considerations

- All images and videos are **fully synthetic** — generated by AI, not sourced from real crime scenes.
- **No real individuals** were used. Prompts avoided identifiable individuals, and outputs with realistic facial features were excluded during quality control.
- **No sensitive personal data** was used in dataset creation.
- All generation was conducted under Google Vertex AI platform content guidelines.
- This dataset is intended for **research purposes only** to support the development of forensic AI tools.

---

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit and cite the paper
- **NonCommercial** — You may not use the material for commercial purposes

---

## Contact

For questions, collaboration, or access requests:
- **Chiamaka Femi-Adeyinka** — cjf068@shsu.edu
- **Cihan Varol** — cxv007@shsu.edu
