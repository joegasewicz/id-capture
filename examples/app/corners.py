import cv2
import numpy as np


class Corners:

    _img_ndarray: np.ndarray
    _img_processed: np.ndarray

    def __init__(self, image_path:str, debug: bool = False):
        self.image_path = image_path
        self.debug = debug

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
        # Resize image to zoom out the detail
        reduce_image = cv2.resize(
            self.img_ndarray,
            None,
            fx=0.25,
            fy=0.25
        )
        # Convert to grey scale
        grey = cv2.cvtColor(reduce_image, cv2.COLOR_BGR2GRAY)
        # Blur
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        # Convert to outline-edges vs non-edges image
        lower_threshold = 50
        upper_threshold = 150
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)


        # Join gaps in any broken edges
        kernel = np.ones((15, 15), np.uint8)
        closed_edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
        )

        if not self.debug:
            self.img_processed = closed_edges
        else:
            self.img_processed = reduce_image.copy()

        contours, _ = cv2.findContours(
            closed_edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        largest_area = cv2.contourArea(contours[0])

        height, width = edges.shape[:2]

        image_area = height * width
        if self.debug:
            for contour in contours[:10]:

                contour_area = cv2.contourArea(contour)
                area_ratio = contour_area / image_area
                if area_ratio < 0.05:
                    continue

                perimeter = cv2.arcLength(
                    contour,
                    True,
                )

                approx = cv2.approxPolyDP(
                    contour,
                    0.02 * perimeter,
                    True,
                )

                # Turn all points in segment & calculate length & direction
                points = approx.reshape(-1, 2)

                segments = []

                for i in range(len(points)):
                    p1 = points[i]
                    p2 = points[(i + 1) % len(points)]

                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]

                    length = np.hypot(dx, dy)

                    angle = np.degrees(
                        np.arctan2(dy, dx)
                    ) % 180

                    segments.append({
                        "p1": p1,
                        "p2": p2,
                        "length": length,
                        "angle": angle,
                    })

                # Compare each segment to the previous.
                for i, segment_a in enumerate(segments):
                    for segment_b in segments[i +1:]:
                        angle_a = segment_a["angle"]
                        angle_b = segment_b["angle"]

                        difference = abs(angle_a - angle_b)
                        difference = min(
                            difference,
                            180 - difference,
                        )

                        if difference > 15:
                            continue

                        length_a = segment_a["length"]
                        length_b = segment_b["length"]

                        length_difference = abs(length_a - length_b)
                        length_ratio = length_difference / max(length_a, length_b)

                        if length_ratio > 0.25:
                            continue

                        if difference <= 15:
                            print(
                                f"Possible opposite sides: "
                                f"{angle_a:.1f} degrees / {angle_b:.1f} | "
                                f"{length_a:.1f}px / {length_b:.1f}px | "
                                f"length difference {length_ratio * 100:.1f}%"
                            )


                print(f"approx points: {len(approx)}")

                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (rect_width, rect_height), angle = rect
                short_side = min(rect_width, rect_height)
                long_side = max(rect_width, rect_height)

                if short_side == 0:
                    continue


                aspect_ratio = long_side / short_side

                print(
                    f"area ratio: {area_ratio:.3f}, "
                    f"aspect ratio: {aspect_ratio:.3f}, "
                    f"angle: {angle:.2f}"
                )

                cv2.drawContours(
                    self.img_processed,
                    [contour],
                    -1,
                    (0, 255, 0),
                    2,
                )

                cv2.drawContours(
                    self.img_processed,
                    [approx],
                    -1,
                    (0, 0, 255),
                    4,
                )

        return False


    def debug_edges(self) -> None:
        # Convert to grey scale
        grey = cv2.cvtColor(self.img_ndarray, cv2.COLOR_BGR2GRAY)
        # Blur
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        # Convert to outline-edges vs non-edges image
        lower_threshold = 50
        upper_threshold = 150
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

        self.img_processed = self.img_ndarray.copy()
        angles = []

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=100,
            maxLineGap=30,
        )
        for line in lines:
            # Each detected line start and end coordinate in an x/y graph.
            x1, y1, x2, y2 = line
            # Debug lines
            cv2.line(
                self.img_processed,
                (x1, y1),
                (x2, y2),
                (0, 255, 0), # Green
                12, # line thickness
            )

    def show(self) -> None:
        print(f"grey shape: {self.img_processed.shape}")
        cv2.imshow("ID Capture", self.img_processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
