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

本项目是论文 **Fine-grained Textual Inversion Network for Zero-Shot Composed Image Retrieval** 的官方实现，发表于 **SIGIR 2024**。

**问题背景：** Composed Image Retrieval（CIR）允许用户通过多模态查询（参考图像 + 修改文本）检索目标图像。然而，现有有监督方法依赖大量标注三元组 `<参考图, 修改文本, 目标图>`，标注代价高昂，数据集规模受限。

**方法核心：** 我们提出了 **FTI4CIR**，无需标注三元组即可完成 CIR 任务。与已有方法将图像映射为单一伪词 token 不同，FTI4CIR 将图像映射为：
- 一个 **subject-oriented 伪词 token**（捕捉图像的主体信息）
- 若干 **attribute-oriented 伪词 tokens**（捕捉主体的多种属性，如袖长、颜色、背景等）

图像由此被表示为句子 `"a photo of [S*] with [A1*, ..., Ar*]"`，进而与修改文本拼接，将 CIR 转化为标准文本-图像检索任务。

**关键组件：**
1. **Fine-grained Pseudo-word Token Mapping**：基于全局特征生成 subject 伪词；通过 Transformer + 局部-全局相关性过滤策略动态提取属性特征，生成 attribute 伪词。
2. **Tri-wise Caption-based Semantic Regularization**：利用 BLIP 生成图像的真实词描述，通过 subject-wise、attribute-wise、whole-wise 三路对齐，将伪词 token 锚定到真实词 embedding 空间。

**实验结果：** 在 FashionIQ、CIRR、CIRCO 三个基准数据集上，FTI4CIR 均取得最优零样本性能。

本仓库提供：完整训练代码、推理与评测代码、预训练模型权重。

---

## Highlights

- 🆕 **首个细粒度文本反转方法**：将图像同时映射为 subject 和 attribute 伪词 token，显著超越单一伪词表示
- 🔧 **动态属性特征提取模块**：自适应处理不同域（时尚、动物、开放域）的多样化局部属性
- 📐 **三路语义正则化**：借助 BLIP 生成的图像描述，对伪词 token 进行 subject / attribute / whole 三级对齐
- 🚫 **完全零样本**：训练仅使用无标注开放域图像，推理无需任何 CIR 标注三元组
- 📊 **三数据集 SOTA**：在 FashionIQ（R@10/50）、CIRR（R@1/5/10/50）、CIRCO 上均优于已有零样本方法

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
├── assets/               # 框架图、结果图
├── src/
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 验证集评测脚本
│   └── test.py           # 测试集预测文件生成脚本
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

### ImageNet

Download ImageNet1K (ILSVRC2012) test set from the [**official site**](https://image-net.org/index.php).

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
│   ├── dress
│   │   └── [B000ALGQSY.jpg | ...]
│   ├── shirt
│   │   └── [B00006M009.jpg | ...]
│   └── toptee
│       └── [B0000DZQD6.jpg | ...]
```

### CIRR

Download from the [**official repository**](https://github.com/Cuberick-Orion/CIRR).

```text
├── CIRR
│   ├── train
│   │   └── [0 | 1 | 2 | ...]
│   │       └── [train-10108-0-img0.png | ...]
│   ├── dev
│   │   └── [dev-0-0-img0.png | ...]
│   ├── test1
│   │   └── [test1-0-0-img0.png | ...]
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
│           └── [000000243611.jpg | ...]
```

---

## Pre-trained Model

Pre-trained model weights: [**Google Drive**](https://drive.google.com/file/d/1WYeBONqZX78IbJ8Jd4qtzZ1zy_ekX-i2/view?usp=sharing)

下载后放入 `model_save/` 目录，或通过 `--model-path` 参数指定路径。

> 图像描述由 [BLIP](https://github.com/salesforce/BLIP) 生成。

---

## Usage

### Training

```bash
python src/train.py \
    --save-frequency 1 \
    --batch-size=256 \
    --lr=4e-5 \
    --wd=0.01 \
    --epochs=60 \
    --model-dir="./model_save" \
    --workers=8 \
    --model ViT-L/14
```

### Validation (split=val)

```bash
python src/evaluate.py \
    --dataset='cirr' \
    --save-path='' \
    --model-path=""
```

| 参数 | 说明 |
|------|------|
| `--dataset` | 数据集，可选 `fashioniq` / `cirr` / `circo` |
| `--model-path` | 预训练模型路径 |
| `--save-path` | 预测结果保存路径 |

### Test (split=test)

生成可上传至官方评测服务器的预测文件：
- [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/)
- [CIRCO Evaluation Server](https://circo.micc.unifi.it/)

```bash
python src/test.py \
    --dataset='cirr' \
    --save-path='' \
    --model-path=""
```

| 参数 | 说明 |
|------|------|
| `--dataset` | 数据集，可选 `cirr` / `circo` |
| `--model-path` | 预训练模型路径 |
| `--save-path` | 预测文件保存路径 |

---

## Citation

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

- [CLIP](https://github.com/openai/CLIP)
- [BLIP](https://github.com/salesforce/BLIP)
- [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq)
- [CIRR](https://github.com/Cuberick-Orion/CIRR)
- [CIRCO](https://github.com/miccunifi/CIRCO)

---

## License

This project is released under the [Apache License 2.0](./LICENSE).
