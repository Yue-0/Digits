__English__ | [简体中文](docs/README_cn.md)

# Handwritten Digit Recognition

## Brief Introduction

This project is the final assignment of my sophomore machine learning course.
This project contents include:

[PyTorch]: https://pytorch.org/ "PyTorch"

* Use [PyTorch] to build [LeNet](https://ieeexplore.ieee.org/document/726791) model;
* Training LeNet model on [MNIST dataset](https://yann.lecun.com/exdb/mnist/);
* Make a handwritten digit recognition applet.

## File Structure

```
Digits
├── data              # Dataset folder
├── docs              # Project Documents
    ├── images        # Images floder
        └── demo.png
    └── README_cn.md  # Chinese description document
├── model
    └── __init__.py
├── install.sh        # Install program
├── LICENSE           # LICENSE
├── main.py           # Handwriting digit recognition application
├── README.md         # English description document
├── requirements.txt  # List of requirements
└── train.py          # Train program
```

## Quick Start

### 1.Clone

```shell
git clone https://github.com/Yue-0/Digits.git
cd ./Digits
```

### 2.Install requirements

```shell
sh install.sh
```

### 3.Train model

```shell
python train.py
```

* The CPU is used for training and deployment by default;
* The training Hyper-parameters are defined in lines 96~98 of [train.py](train.py):
```python
LR = 1e-4   # learning rate
EPOCH = 20  # epochs
BATCH = 64  # batch size
```

### 4.Run the handwritten digit recognition application

```shell
python main.py
```

After the program runs, a whiteboard will appear.
You can press and hold the left mouse button to write on the whiteboard.
After releasing the left mouse button, the program will automatically recognize
the numbers on the whiteboard and display the recognition results below the
whiteboard, as shown in the following figure
(_Currently, only one number can be recognized_)：

![demo](docs/images/demo.png)

You can click the right mouse button to clear the whiteboard.
