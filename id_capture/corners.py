from typing import Optional

import cv2
import numpy as np
from cv2 import Mat
from numpy import ndarray, dtypes

from id_capture.draw import Draw


class Corners:

    _img_ndarray: np.ndarray
    _img_processed: Optional[np.ndarray] = None

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

    def contains_card(self) -> bool:
        """
        Threshold: Cut off brightness value:
            - pixels above it turn white (255).
            - pixels below it turn white (0).

        Contour: Contours are the outlines of a shape:

            array([[[120,  45]],
            [[480,  50]],
            [[478, 300]],
            [[118, 295]]])   # e.g. 4 corners of a card

        :return:
        """
        img = self._reduce_image_size(self.img_ndarray)
        img = self._convert_to_grey_scale(img)
        img = self._blur_image(img)
        img = self._find_edges(img)
        img = self._fill_line_gaps(img)

        contours = self._get_contours(img)

        if not contours:
            return False

        # Create the cleaned mask - turns think outline into a solid shape.
        mask = self._create_filled_mask(img, contours[0])
        cleaned_mask = self._remove_protrusions(mask)
        # Create the contours (Outlines of the shape)
        contours = self._get_contours(cleaned_mask)

        if not contours:
            return False

        self.img_processed = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        image_area = self._get_image_area(img)
        segments = self._extract_edge_segments(contours, image_area)
        approximates = self._analyse_card_candidates(contours, segments, image_area)

        if self.debug:
            draw = Draw(self.img_processed)
            draw.draw_contours([approx["contour"] for approx in approximates], (0, 255, 0))
            # draw.draw_contours([approx["approx"] for approx in approximates], (0, 0, 255))

        return self._is_card_candidate(approximates)

    def _is_card_candidate(self, approximates) -> bool:
        for candidate in approximates:
            if candidate["is_card"]:
                return True
        return False

    def show(self) -> None:
        if self.img_processed is None:
            self.img_processed = self.img_ndarray.copy()
        cv2.imshow("ID Capture", self.img_processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _reduce_image_size(self, img: ndarray) -> Mat | ndarray:
        """
        Resize image to zoom out the detail
        :return:
        """
        return cv2.resize(
            img,
            None,
            fx=0.25,
            fy=0.25
        )

    def _convert_to_grey_scale(self, img: ndarray) -> Mat | ndarray:
        """
        Reduce to a single channel to make thresholding, edge detection
        & contour finding faster & simpler. Boundaries in open CV
        are found from changes in brightness, not color.
        :param img:
        :return:
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _blur_image(self, img: ndarray) -> Mat | ndarray:
        """
        Blur the image to smooth out noise & fine texture so edge detection
        & thresholding pick's up the real outline.
        :param img:
        :return:
        """
        return cv2.GaussianBlur(img, (5, 5), 0)

    def _find_edges(self, img: ndarray) -> Mat | ndarray:
        """
        All brightness, shading & texture are removed from the image.
        Each pixel is either fully black or fully white.
        :param img:
        :return:
        """
        lower_threshold = 50
        upper_threshold = 150
        edges = cv2.Canny(img, lower_threshold, upper_threshold)
        return edges

    def _fill_line_gaps(self, img: ndarray) -> Mat | ndarray:
        """
        Joins broken white edge lines into complete outlines.
        :return:
        """
        # set the area around the line pixel to test
        kernel = np.ones((5, 5), np.uint8)
        # With RETR_EXTERNAL can trace the card's edge as one closed shape instead of many broken pieces.
        closed_edges = cv2.morphologyEx(
            img,
            cv2.MORPH_CLOSE,
            kernel,
        )
        return closed_edges

    def _get_contours(self, img: ndarray) -> list:
        """
        Turns image into a list of shapes, sorted large to small.
        :param img:
        :return:
        """
        contours, _ = cv2.findContours(
            img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )
        return contours

    def _get_image_area(self, img: ndarray) -> int:
        height, width = img.shape[:2]
        image_area = height * width
        return image_area

    def _analyse_card_candidates(
        self,
        contours: list,
        segments: list,
        image_area: int,
    ) -> list[dict]:
        """
        1. Simplifies the outline to its corners with approxPolyDP.
        2. Finds possible opposite sides.
        3. Measures the shape if it fits a rotated rectangle.

        :param contours:
        :param segments:
        :param image_area:
        :return:
        """
        approximates = []

        for contour in contours[:10]:
            # Compare each segment to the previous.
            contour_area = cv2.contourArea(contour)
            area_ratio = contour_area / image_area
            if area_ratio < 0.05:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

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

            rect = cv2.minAreaRect(contour)
            _, (rect_width, rect_height), angle = rect

            short_side = min(rect_width, rect_height)
            long_side = max(rect_width, rect_height)

            if short_side == 0:
                continue

            aspect_ratio = long_side / short_side
            rectangle_area = rect_width * rect_height

            if rectangle_area == 0:
                continue

            rectangularity = contour_area / rectangle_area
            has_four_corners = len(approx) == 4
            is_convex = cv2.isContourConvex(approx)

            candidate_dict = {
                "has_four_corners": has_four_corners,
                "contour": contour,
                "approx": approx,
                "area_ratio": area_ratio,
                "aspect_ratio": aspect_ratio,
                "rectangularity": rectangularity,
                "is_card": False,
                "is_convex": is_convex,
            }

            candidate_dict["is_card"] = self._is_card(candidate_dict)

            print({
                "corners": len(approx),
                "convex": is_convex,
                "aspect_ratio": aspect_ratio,
                "rectangularity": rectangularity,
                "area_ratio": area_ratio,
            })

            approximates.append(candidate_dict)
        return approximates

    def _extract_edge_segments(self, contours: list, image_area: int) -> list:
        """
        1. Simplifies the outline of a polygon's corners.
        2. Splits it into straight sides (recording each side's length & direction).
        This is the data required to test if opposite sides look like a card shape.
            - e.g. parallel & roughly the same length.
        :param contours:
        :param image_area:
        :return:
        """
        segments = []
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
        return segments

    def _find_straight_lines(self, edges: ndarray) -> Optional[ndarray]:
        height, width = edges.shape[:2]

        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=int(width * 0.15),
            maxLineGap=int(width * 0.10),
        )

    def _create_filled_mask(self, img: ndarray, contour: ndarray) -> ndarray:
        """
        Draws the contour as a solid shape on a black canvas the same
        size as the image.
        :param img:
        :param contour:
        :return:
        """
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        return mask

    def _remove_protrusions(self, mask:ndarray, kernel_size: int = 21) -> ndarray:
        """
        Strips thin parts that stick out of the shape, e.g. finders holding
        the card, while keeping the main body.
        :param mask:
        :param kernel_size:
        :return:
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _is_card(self, candidate: dict) -> bool:
        return (
                candidate["has_four_corners"]
                and candidate["is_convex"]
                and 1.35 <= candidate["aspect_ratio"] <= 1.85
                and candidate["rectangularity"] >= 0.75
                and candidate["area_ratio"] >= 0.05
        )
