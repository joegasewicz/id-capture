import cv2
import numpy as np


class Corners:

    _img_ndarray: np.ndarray

    def __init__(self, image_path:str):
        self.image_path = image_path

    @property
    def img_ndarray(self) -> np.ndarray:
        return self._img_ndarray

    @img_ndarray.setter
    def img_ndarray(self, value: np.ndarray) -> None:
        self._img_ndarray = value

    def load_image(self) -> None:
        self.img_ndarray = cv2.imread(self.image_path)

    def show(self) -> None:
        grey = cv2.cvtColor(self.img_ndarray, cv2.COLOR_BGR2GRAY)
        print(f"grey shape: {grey.shape}")
        cv2.imshow("ID Capture", grey)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
