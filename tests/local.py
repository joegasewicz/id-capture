from id_capture.corners import Corners


def make_app():
    print(f"Starting...")
    corners = Corners("data/CS02_29.tif", debug=False)
    corners.load_image()
    is_card = corners.contains_card()
    corners.show()

    # print(f"Does a card exist in the image: {is_card}")

if __name__ == "__main__":
    make_app()
