__English__ | [简体中文](docs/README_cn.md)

# Handwritten Digit Recognition

## Brief Introduction

This project is an assignment for my undergraduate machine learning course. 
The project content includes:

[LeNet5]: https://ieeexplore.ieee.org/document/726791 "LeNet5"

* Use [PyTorch](https://pytorch.org/) to build [LeNet5] model;
* Training [LeNet5] model on [MNIST dataset](https://yann.lecun.com/exdb/mnist/);
* Make a handwritten digit recognition applet.

## Demo

After running the main program, a whiteboard will appear. 
Press and hold the left mouse button to write on the whiteboard. 
After releasing the left mouse button, the program will automatically 
recognize the numbers on the whiteboard and display the recognition results 
below the whiteboard. Press the right button of the mouse to clear the whiteboard.

![Demo](docs/demo.gif)

## File Structure

```
Digits
├── docs              # Project Documents
    ├── demo.gif      # Demo
    └── README_cn.md  # Chinese Description Document
├── model             # LeNet Package
    └── __init__.py   # LeNet Model
├── install.sh        # Install program
├── LICENSE           # LICENSE
├── main.py           # Handwriting Digit Recognition Application
├── README.md         # English Description Document
├── requirements.txt  # List of Requirements
└── train.py          # Training Program
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

Due to the simple network structure, 
this project uses CPU for training and inference.

### 4.Run the handwritten digit recognition application

```shell
python main.py
```
