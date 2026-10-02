import csv

import cv2

from app.corners import Corners


def make_app():
    print(f"Starting...")
    corners = Corners("data/88.jpg", debug=True)
    corners.load_image()
    corners.is_card()
    # corners.debug_edges()
    corners.show()
