import cv2
import numpy as np


class Corners:

    _img_ndarray: np.ndarray
    _img_processed: np.ndarray

    def __init__(self, image_path:str):
        self.image_path = image_path

    @property
    def img_ndarray(self) -> np.ndarray:
        return self._img_ndarray

    @img_ndarray.setter
    def img_ndarray(self, value: np.ndarray) -> None:
        self._img_ndarray = value

    @property
    def img_processed(self) -> np.ndarray:
        return self._img_processed

    @img_processed.setter
    def img_processed(self, value: np.ndarray) -> None:
        self._img_processed = value

    def load_image(self) -> None:
        self.img_ndarray = cv2.imread(self.image_path)

    def is_card(self) -> bool:
        # Convert to grey scale
        grey = cv2.cvtColor(self.img_ndarray, cv2.COLOR_BGR2GRAY)
        # Blur
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        # Convert to outline-edges vs non-edges image
        lower_threshold = 50
        upper_threshold = 150
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

        self.img_processed = edges

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=100,
            maxLineGap=30,
        )
        if lines is None:
            return False

        # Get angle for each detected line
        angles = []
        for line in lines:
            # Each detected line start and end coordinate in an x/y graph.
            x1, y1, x2, y2 = line
            # Get angle in degrees
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle %= 180
            angles.append(angle)
            # print(f"{line[0]} -> {angle:.2f} degrees")

        # Check lines are perpendicular to each other.
        angle_tolerance = 10
        for angle_a in angles:
            for angle_b in angles:
                difference = abs(angle_a - angle_b)
                difference = min(difference, 180 - difference)

                if abs(difference - 90) <= angle_tolerance:
                    print(
                        f"Possible card edges:"
                        f"{angle_a:.2f} degrees & {angle_b:.2f}"
                    )

        return False




    def show(self) -> None:
        print(f"grey shape: {self.img_processed.shape}")
        cv2.imshow("ID Capture", self.img_processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
