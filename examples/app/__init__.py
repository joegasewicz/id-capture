import csv

import cv2

from app.corners import Corners


def make_app():
    print(f"Starting...")
    corners = Corners("data/no-id.jpg")
    corners.load_image()
    # corners.is_card()
    corners.debug_edges()
    corners.show()
