# Fine-grained Textual Inversion Network for Zero-Shot Composed Image Retrieval

> Mapping reference images into subject-oriented and attribute-oriented pseudo-word tokens for zero-shot composed image retrieval, without requiring any annotated training triplets.

## Authors

**Haoqiang Lin**<sup>1</sup>, **Haokun Wen**<sup>2</sup>, **Xuemeng Song**<sup>1</sup>\*, **Meng Liu**<sup>3</sup>, **Yupeng Hu**<sup>1</sup>, **Liqiang Nie**<sup>2</sup>

<sup>1</sup> Shandong University, Qingdao / Jinan, China  
<sup>2</sup> Harbin Institute of Technology (Shenzhen), Shenzhen, China  
<sup>3</sup> Shandong Jianzhu University, Jinan, China  
\* Corresponding author: sxmustc@gmail.com

## Links

- **Paper**: [`SIGIR '24`](https://doi.org/10.1145/3626772.3657831)
- **Pre-trained Model**: [`Google Drive`](https://drive.google.com/file/d/1WYeBONqZX78IbJ8Jd4qtzZ1zy_ekX-i2/view?usp=sharing)
- **Code Repository**: [`GitHub`](https://github.com/ZiChao111/FTI4CIR)

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset / Benchmark](#dataset--benchmark)
- [Pre-trained Model](#pre-trained-model)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [07/2024] Paper accepted and presented at SIGIR 2024 (Washington, DC, USA)
- [07/2024] Initial code release

---

## Introduction

This repository contains the official implementation of the paper **Fine-grained Textual Inversion Network for Zero-Shot Composed Image Retrieval**, published at **SIGIR 2024**.

**Background.** Composed Image Retrieval (CIR) enables users to retrieve target images using a multimodal query consisting of a reference image and a modification text describing the desired changes. However, existing supervised CIR methods rely heavily on expensive annotated training triplets in the form of `<reference image, modification text, target image>`, which limits dataset scale and model generalization.

**Method.** We propose **FTI4CIR**, which addresses CIR without requiring any annotated triplets. Unlike prior methods that map an image to a single coarse-grained pseudo-word token, FTI4CIR maps each image into:
- One **subject-oriented pseudo-word token** — capturing the primary subject(s) of the image.
- Several **attribute-oriented pseudo-word tokens** — capturing contextual attributes such as sleeve length, color, and background.

Each image is represented by the sentence `"a photo of [S*] with [A1*, ..., Ar*]"`, which is concatenated with the modification text to form a unified text query, reducing CIR to a standard text-to-image retrieval task.

**Key Components.**
1. **Fine-grained Pseudo-word Token Mapping**: Generates subject pseudo-words from global image features, and leverages a Transformer with a local-global relevance-based filtering strategy to dynamically extract attribute-oriented features.
2. **Tri-wise Caption-based Semantic Regularization**: Employs BLIP to generate natural language descriptions and aligns pseudo-word tokens to the real-word embedding space via three complementary objectives: subject-wise, attribute-wise, and whole-wise.

**Results.** FTI4CIR achieves state-of-the-art zero-shot performance on FashionIQ, CIRR, and CIRCO.

This repository provides: full training code, inference and evaluation scripts, and pre-trained model weights.

---

## Highlights

- 🆕 **First fine-grained textual inversion approach**: Maps images into both subject-oriented and attribute-oriented pseudo-word tokens, surpassing single pseudo-word representations.
- 🔧 **Dynamic local attribute feature extraction**: Adaptively handles diverse attribute types across different domains (fashion, animals, open-domain).
- 📐 **Tri-wise semantic regularization**: Aligns pseudo-word tokens to the real-word embedding space via subject-wise, attribute-wise, and whole-wise objectives using BLIP-generated captions.
- 🚫 **Fully zero-shot**: Training relies solely on unlabeled open-domain images; no annotated CIR triplets are required at any stage.
- 📊 **State-of-the-art on three benchmarks**: Outperforms all zero-shot baselines on FashionIQ (R@10/50), CIRR (R@1/5/10/50), and CIRCO.

---

## Method / Framework

<img width="1662" height="524" alt="image" src="https://github.com/user-attachments/assets/b77761fd-8c5a-4709-845e-2b24b253c4a7" />

**Figure 1.** Overall framework of FTI4CIR, consisting of (a) Fine-grained Pseudo-word Token Mapping and (b) Tri-wise Caption-based Semantic Regularization.

### Inference Pipeline

```
Reference Image
      │
      ▼
Fine-grained Textual Inversion
      │
      ▼
"A photo of [S*] with [A1*, ..., Ar*]."
      │
      + Modification Text
      │
      ▼
Pure Text Query → Text-to-Image Retrieval → Target Image
```

---

## Project Structure

```text
.
├── assets/               # Figures and visualizations
├── src/
│   ├── train.py          # Training script
│   ├── evaluate.py       # Validation evaluation script
│   └── test.py           # Test-split prediction generation script
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ZiChao111/FTI4CIR.git
cd FTI4CIR
```

### 2. Running Environment

```
Platform : NVIDIA A100 40G
Python   : 3.9.12
PyTorch  : 2.2.0
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset / Benchmark

Please download each dataset following the instructions in the respective official repositories and organize them as described below.

### ImageNet

Download the ImageNet1K (ILSVRC2012) test set from the [**official site**](https://image-net.org/index.php).

```text
├── ImageNet
│   └── test
│       └── [ILSVRC2012_test_00000001.JPEG | ... | ILSVRC2012_test_00100000.JPEG]
```

### FashionIQ

Download from the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq).

```text
├── FashionIQ
│   ├── captions
│   │   ├── cap.dress.[train | val | test].json
│   │   ├── cap.toptee.[train | val | test].json
│   │   └── cap.shirt.[train | val | test].json
│   ├── image_splits
│   │   ├── split.dress.[train | val | test].json
│   │   ├── split.toptee.[train | val | test].json
│   │   └── split.shirt.[train | val | test].json
│   ├── dress / shirt / toptee
│   │   └── [*.jpg]
```

### CIRR

Download from the [**official repository**](https://github.com/Cuberick-Orion/CIRR).

```text
├── CIRR
│   ├── train / dev / test1
│   │   └── [*.png]
│   └── cirr
│       ├── captions
│       │   └── cap.rc2.[train | val | test1].json
│       └── image_splits
│           └── split.rc2.[train | val | test1].json
```

### CIRCO

Download from the [**official repository**](https://github.com/miccunifi/CIRCO).

```text
├── CIRCO
│   ├── annotations
│   │   └── [val | test].json
│   └── COCO2017_unlabeled
│       ├── annotations
│       │   └── image_info_unlabeled2017.json
│       └── unlabeled2017
│           └── [*.jpg]
```

---

## Pre-trained Model

Pre-trained model weights are available on [**Google Drive**](https://drive.google.com/file/d/1WYeBONqZX78IbJ8Jd4qtzZ1zy_ekX-i2/view?usp=sharing).

After downloading, place the checkpoint under `model_save/`, or specify the path via the `--model_path` argument at evaluation time.

> Image captions used during training are generated by [BLIP](https://github.com/salesforce/BLIP). Please refer to the BLIP repository for caption generation details.

---

## Usage

### Training

```bash
python src/train.py \
    --save_frequency 1 \
    --batch_size=256 \
    --lr=4e-5 \
    --wd=0.01 \
    --epochs=60 \
    --model_dir="./model_save" \
    --workers=8 \
    --model ViT-L/14
```

### Validation (split=val)

```bash
python src/evaluate.py \
    --dataset='cirr' \
    --save_path='' \
    --model_path=""
```

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset to evaluate on. Options: `fashioniq` / `cirr` / `circo` |
| `--model_path` | Path to the pre-trained model checkpoint |
| `--save_path` | Path to save the prediction results |

### Test (split=test)

Generate prediction files for submission to the official evaluation servers:
- [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/)
- [CIRCO Evaluation Server](https://circo.micc.unifi.it/)

```bash
python src/test.py \
    --dataset='cirr' \
    --save_path='' \
    --model_path=""
```

| Argument | Description |
|----------|-------------|
| `--dataset` | Options: `cirr` / `circo` |
| `--model_path` | Path to the pre-trained model checkpoint |
| `--save_path` | Path to save the prediction file |

---

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{FTI4CIR,
  author    = {Haoqiang Lin and Haokun Wen and Xuemeng Song and Meng Liu and Yupeng Hu and Liqiang Nie},
  title     = {Fine-grained Textual Inversion Network for Zero-Shot Composed Image Retrieval},
  booktitle = {Proceedings of the International {ACM} SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {240--250},
  publisher = {{ACM}},
  year      = {2024}
}
```

---

## Acknowledgement

We thank the following open-source projects for providing valuable components and implementations:

- [CLIP](https://github.com/openai/CLIP)
- [BLIP](https://github.com/salesforce/BLIP)
- [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq)
- [CIRR](https://github.com/Cuberick-Orion/CIRR)
- [CIRCO](https://github.com/miccunifi/CIRCO)

---

## License

This project is released under the [Apache License 2.0](./LICENSE).
