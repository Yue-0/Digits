import cv2
import torch
import numpy as np

from model import LeNet


QUIT, WAIT, DRAW, SHOW = tuple(range(4))


class Window:
    def __init__(self, name: str, size: int = 640):
        """
        :param name: Window name.
        :param size: Window size.
        """
        self.num = ()
        self.run = WAIT
        self.size = size
        self.name = name
        cv2.namedWindow(name)
        self.scale = size >> 5
        self.image = self.clear()
        self.classifier = LeNet()
        cv2.setMouseCallback(name, self.callback)
        self.classifier.load_state_dict(torch.load("model/LeNet.pt"))

    def show(self) -> None:
        """
        Program main loop.
        """
        self.classifier.eval()
        while self.run:
            image = cv2.copyMakeBorder(self.image, 0, self.size >> 4, 0, 0,
                                       cv2.BORDER_CONSTANT, value=255)
            if self.run == SHOW:
                cv2.putText(
                    image, "result: {}, similarity: {}%".format(
                        self.num[0], self.num[1]
                    ), (0, self.size + 32), cv2.FONT_HERSHEY_TRIPLEX, 1, 0, 1
                )
            cv2.imshow(self.name, image)
            cv2.waitKey(WAIT)
            try:
                cv2.getWindowProperty(self.name, cv2.WND_PROP_AUTOSIZE)
            except cv2.error:
                self.run = QUIT
                cv2.destroyAllWindows()

    def clear(self) -> np.ndarray:
        """
        return a whiteboard image.
        """
        return ~np.zeros((self.size, self.size), np.uint8)

    def callback(self, event, x, y, *params):
        """
        OpenCV mouse callback function.
        """
        assert len(params) == 2
        if event == cv2.EVENT_RBUTTONUP:
            self.run = WAIT
            self.image = self.clear()
        if event == cv2.EVENT_LBUTTONUP:
            self.num = self.classification()
            self.run = SHOW
        if event == cv2.EVENT_LBUTTONDOWN:
            self.run = DRAW
        if self.run == DRAW and max(x, y) < self.size:
            cv2.circle(self.image, (x, y), self.scale, 0, -1)

    def classification(self,
                       size: tuple[int, int] = (32, 32)) -> tuple[int, float]:
        """
        Identify the image, return identification result and similarity.
        """
        prob = np.exp(self.classifier(torch.reshape(torch.tensor(cv2.threshold(
            cv2.resize(self.image, size), 127, 1, cv2.THRESH_BINARY_INV
        )[1], dtype=torch.float32), (1, 1) + size))[0].detach().numpy())
        argmax = np.argmax(prob)
        return argmax, int(100 * prob[argmax] / prob.sum())


if __name__ == "__main__":
    Window("Digits").show()
