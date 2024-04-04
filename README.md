# FTI4CIR

This is an open-source implementation of the paper "Fine-grained Textual Inversion network for Zero-Shot Composed Image Retrieval" (**FTI4CIR**).

*Tip: We will gradually upload the organized code in the future.*


### Installation
1. Clone the repository

```sh
git clone https://github.com/ZiChao111/FTI4CIR.git
```

2. Install Python dependencies

```sh
conda create -n FTI4CIR -y python=3.9
conda activate FTI4CIR

```


### Data Preparation

#### ImageNet

Download ImageNet1K (ILSVRC2012) test set following the instructions in
the [**official site**](https://image-net.org/index.php).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── ImageNet1K
│   ├── test
|   |   ├── [ILSVRC2012_test_[00000001 | ... | 00100000].JPEG]
```

#### FashionIQ

Download the FashionIQ dataset following the instructions in
the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── FashionIQ
│   ├── captions
|   |   ├── cap.dress.[train | val | test].json
|   |   ├── cap.toptee.[train | val | test].json
|   |   ├── cap.shirt.[train | val | test].json

│   ├── image_splits
|   |   ├── split.dress.[train | val | test].json
|   |   ├── split.toptee.[train | val | test].json
|   |   ├── split.shirt.[train | val | test].json

│   ├── dress
|   |   ├── [B000ALGQSY.jpg | B000AY2892.jpg | B000AYI3L4.jpg |...]

│   ├── shirt
|   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]

│   ├── toptee
|   |   ├── [B0000DZQD6.jpg | B000A33FTU.jpg | B000AS2OVA.jpg | ...]
```

#### CIRR

Download the CIRR dataset following the instructions in the [**official repository**](https://github.com/Cuberick-Orion/CIRR).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRR
│   ├── train
|   |   ├── [0 | 1 | 2 | ...]
|   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

│   ├── dev
|   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

│   ├── cirr
|   |   ├── captions
|   |   |   ├── cap.rc2.[train | val | test1].json
|   |   ├── image_splits
|   |   |   ├── split.rc2.[train | val | test1].json
```

#### CIRCO

Download the CIRCO dataset following the instructions in the [**official repository**](https://github.com/miccunifi/CIRCO).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRCO
│   ├── annotations
|   |   ├── [val | test].json

│   ├── COCO2017_unlabeled
|   |   ├── annotations
|   |   |   ├──  image_info_unlabeled2017.json
|   |   ├── unlabeled2017
|   |   |   ├── [000000243611.jpg | 000000535009.jpg | ...]
```



### Caption Generation



### Pre-training Phase

### Sample running code for training:

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

### Inference Phase

#### Validation (split=val)

Evaluation on FashionIQ, CIRR, or CIRCO.

```sh
python src/evaluate.py \
    --dataset='cirr' \
    --save-path='' \
    --model-path="" \
    --CIRR-path="" \
    --CIRCO-path="" \
```

```
    --dataset <str>                 Dataset to use, options: ['cirr', 'circo']
    --CIRR-path <str>               Path to the CIRR dataset root folder
    --CIRCO-path <str>              Path to the CIRCO dataset root folder
    --model-path <str>              Path of the pre-trained model
    --save-path <str>               Path to save the predictions file
```


</details>

#### Test (split=test)

To generate the predictions file for uploading on the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) or the [CIRCO Evaluation Server](https://circo.micc.unifi.it/) using the our model, please execute the following command:

```sh
python src/test.py \
    --dataset='cirr' \
    --save-path='' \
    --model-path="" \
    --CIRR-path="" \
    --CIRCO-path="" \
```

```
    --dataset <str>                 Dataset to use, options: ['cirr', 'circo']
    --CIRR-path <str>               Path to the CIRR dataset root folder
    --CIRCO-path <str>              Path to the CIRCO dataset root folder
    --model-path <str>              Path of the pre-trained model
    --save-path <str>               Path to save the predictions file
```


</details>




### Acknowledgement



