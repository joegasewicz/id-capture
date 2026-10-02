import csv

import cv2

from app.corners import Corners


def make_app():
    print(f"Starting...")
    corners = Corners("data/88.jpg")
    img_arr = corners.load_image()
    print(img_arr.shape)
    cv2.imshow("ID Capture: ", img_arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
