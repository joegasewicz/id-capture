import cv2
import numpy as np


class Corners:

    def __init__(self, image_path:str):
        self.image_path = image_path

    def load_image(self) -> np.ndarray:
        return cv2.imread(self.image_path)
