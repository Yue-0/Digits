from time import time
from os.path import join
from random import randint
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
from torch.utils import data
from torchvision import datasets

from model import LeNet


class MNIST(data.Dataset):
    def __init__(self, mode: str):
        try:
            self.train = mode == "train"
            self.dataset = datasets.MNIST("data", self.train)
        except RuntimeError:
            raise FileNotFoundError(
                "Dataset not found, please execute \"sh install.sh\""
            )
        assert mode in ("train", "test"), "mode must be 'train' or 'test'"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item) -> tuple:
        image, label = self.dataset[item]
        image = np.array(image, np.uint8)
        if self.train:
            size = randint(7, 16)
            image = cv2.resize(image, (size << 1,) * 2)
            if size < 16:
                size = 16 - size
                image = cv2.copyMakeBorder(
                    image, size, size, size, size, cv2.BORDER_CONSTANT, value=0
                )
        else:
            image = cv2.resize(image, (32, 32))
        return torch.unsqueeze(torch.tensor(
            cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1],
            dtype=torch.float32
        ), 0), label


class Model:
    def __init__(self, lr: float):
        self.lr = lr
        self.net = LeNet
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr)

    def fit(self, train: MNIST, test: MNIST, batch: int, epoch: int) -> None:
        """
        Train model.
        :param train: Training dataset.
        :param test: Validation dataset.
        :param batch: Batch size.
        :param epoch: Training epoch.
        """
        best, dataset = 0, data.DataLoader(train, batch, True)
        for e in range(epoch):
            self.net.train()
            t, err = time(), 0
            print(f"Epoch: {e + 1}/{epoch}")
            for b, (images, labels) in enumerate(dataset):
                loss = self.loss(self.net(images), labels)
                loss.backward()
                err += loss.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                num = min((batch * (b + 1)), len(train))
                speed, progress = num / (time() - t + 1e-9), num / len(train)
                print("\rTrain: [{}>{}]{}%  {}images/s  loss: {:.2f} ".format(
                    "=" * int(25 * progress), "." * (25 - int(25 * progress)),
                    round(100 * progress), round(speed), err / (b + 1)
                ), end=f"eta {round((len(train) - num) / speed)}s")
            print()
            acc = self.score(test, batch)
            if acc > best:
                best = acc
                torch.save(self.net.state_dict(), join("model", "LeNet.pt"))
        print(f"\nBest acc: {100 * best}%, model is saved in 'model/LeNet.pt'")

    def score(self, test: MNIST, batch: int) -> float:
        """
        Test model.
        :param test: Validation dataset.
        :param batch: Batch size.
        :return: Accuracy of the model on the validation dataset.
        """
        self.net.eval()
        t, acc = time(), 0
        for b, (images, labels) in enumerate(data.DataLoader(test, batch)):
            num = min((batch * (b + 1)), len(test))
            acc += np.int32(torch.argmax(self.net(images), 1) == labels).sum()
            speed, progress = num / (time() - t + 1e-9), num / len(test)
            print("\r Eval: [{}>{}]{}%  {}images/s   acc: {:.2f}% ".format(
                "=" * int(25 * progress), "." * (25 - int(25 * progress)),
                round(100 * progress), round(speed), 100 * acc / num
            ), end=f"eta {round((len(test) - num) / speed)}s")
        print()
        return acc / len(test)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "--batch", type=int, default=64, help="Batch size. Default: 64"
    )
    args.add_argument(
        "--epoch", type=int, default=10, help="Training epoch. Default: 10"
    )
    args.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate. Default: 0.1"
    )
    args = args.parse_args()
    Model(args.lr).fit(MNIST("train"), MNIST("test"), args.batch, args.epoch)
