# Human Face Recogniton System

Phan Quốc Tuấn 

## Contents
[Face Recognition](#face-recognition)
- [Introduction](#introduction)
- [Installation](#installation)
- [Pre-trained](#)
- [Run a demo](#run-a-demo)

[Face Regconition between Python client and Python server]

[Face Regcontion between Jave client and Python server]

## Face Recognition

### Introduction

This is an face recognition system was built base on pipeline:

<img src="https://imgur.com/4Fhhsj1">

- For Face Detection, my method use [MTCNN](https://github.com/ipazc/mtcnn) with minisize=50.
- For Feature Extraction, my method use [Arcface](https://github.com/deepinsight/insightface) with pre-trained [LResNet100E-IR,ArcFace@ms1m-refine-v2](https://github.com/deepinsight/insightface/wiki/Model-Zoo).
- For Feature Retrieval, my method using [Faiss](https://github.com/facebookresearch/faiss).

### Installation

1. Use Anaconda for create virtual environment

```
conda create -n FaceReg python=3.7
```

2. Clone the repository

```
git clone
```

3. Install 'MXNet' with GPU support on cuda-10.1 (Python 3.7)

```
pip install mxnet-cu101
```

(If you only use CPU, use following command instead)

```
pip install mxnet-cpu
```

4. Install some required library

```
pip install scikit-learn
pip install scilit-image
pip install easydict
```

### Download a pre-trained

Download a pre-trained model for Arcface at [LResNet100E-IR,ArcFace@ms1m-refine-v2](https://github.com/deepinsight/insightface/wiki/Model-Zoo). and put it in ./model/

### Run a demo

1. Prepare your recognition data in ./deploy/Images folder

deploy
+--_Images
|  +--A
|  |   +--A_1.jpg
|  |   +--A_2.jpg
|  |   +--...
|  +--B
|  |   +--B_1.jpg
|  |   +--...
|  +--...

2. Feature Extraction of all images data in Images folder and write to ./deploy/Data

(Before run make sure you in ./deploy directory)

```
python pre_process.py
```

3. Let's run your demo!

```
python faiss_reg.py
```




