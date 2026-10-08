import cv2
from numpy import ndarray



class Draw:

    def __init__(self, img: ndarray):
        self.img = img

    def draw_contours(self, contours: list, color: tuple) -> None:
        for contour in contours:
            cv2.drawContours(
                self.img,
                [contour],
                -1,
                color,
                4,
            )
