import os

import cv2
import numpy as np
from PIL import Image

from detector import Detector
from letter_model import LetterModel

if __name__ == "__main__":
    d = Detector("../yolov5", "../yolov5/runs/train/exp5/weights/best.pt")
    lm = LetterModel("model_weights.pth")

    samples = []
    root = "../figure-generator/generated/test/"
    for _, _, files in os.walk(root):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith(".png"):
                samples.append(path)

    img = Image.open(samples[0])

    # YOLO only
    # results = d.detect_figures(img)

    # YOLO + letters
    results = d.detect_with_letters(img, lm)

    for r in results:
        print("box:", r["box"])
        print("conf:", r["conf"])
        print("cls:", r["cls"])
        print("label:", r["label"])
        if "letter" in r:
            print("letter:", r["letter"])

        cv2.imshow("test", np.asarray(r["im"]))
        cv2.waitKey(1000)
