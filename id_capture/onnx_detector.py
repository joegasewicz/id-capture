"""Runtime-only adapter for the decoded YOLOX ONNX export from notebook 10.

No PyTorch, YOLOX source, or Ultralytics imports are needed at runtime.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

CONTRACT_KEY = "id_capture.contract"
CLASS_NAMES = ("driving_licence", "passport")


def preprocess_bgr(image, size):
    """Top-left letterbox, BGR float32 NCHW, 0..255, padding 114."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or min(image.shape[:2]) < 1):
        raise ValueError("Expected a nonempty HWC BGR uint8 image.")
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width, resized_height = int(width * ratio), int(height * ratio)
    if min(resized_width, resized_height) < 1:
        raise ValueError("Image aspect ratio is too extreme for this input size.")
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, ratio


def read_bgr(path):
    """Coordinates returned by the detector refer to this EXIF-upright image."""
    with Image.open(path) as original:
        rgb = np.array(ImageOps.exif_transpose(original).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _class_nms(boxes, scores, classes, threshold):
    # Match YOLOX's class-aware NMS; preserve high-score detections first.
    remaining = np.argsort(-scores, kind="stable")
    keep = []
    while remaining.size:
        index = int(remaining[0])
        keep.append(index)
        remaining = remaining[1:]
        if not remaining.size:
            break
        overlap = np.maximum(0, np.minimum(boxes[index, 2:], boxes[remaining, 2:])
                             - np.maximum(boxes[index, :2], boxes[remaining, :2])).prod(axis=1)
        area = np.maximum(0, boxes[index, 2:] - boxes[index, :2]).prod()
        areas = np.maximum(0, boxes[remaining, 2:] - boxes[remaining, :2]).prod(axis=1)
        iou = overlap / np.maximum(area + areas - overlap, np.finfo(np.float32).eps)
        remaining = remaining[(classes[remaining] != classes[index]) | (iou <= threshold)]
    return np.array(keep, dtype=np.int64)


def decode_predictions(output, ratio, width, height, confidence=0.2, nms_iou=0.65):
    """Decoded [1,N,7] cx,cy,w,h,obj,p_licence,p_passport -> Nx6 xyxy,score,class.

    There is no sigmoid or grid decoding here: those operations are in the graph.
    NMS happens before clipping boxes, matching the training evaluation helper.
    """
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 7:
        raise ValueError(f"Expected decoded YOLOX [1,N,7], got {output.shape}.")
    if not (np.isfinite(ratio) and ratio > 0 and width > 0 and height > 0
            and np.isfinite(confidence) and 0 <= confidence <= 1
            and np.isfinite(nms_iou) and 0 <= nms_iou <= 1):
        raise ValueError("Invalid geometry or detection thresholds.")
    rows = output[0]
    classes = rows[:, 5:].argmax(axis=1)
    scores = rows[:, 4] * rows[np.arange(len(rows)), 5 + classes]
    mask = (np.isfinite(rows).all(axis=1) & (scores >= confidence)
            & (rows[:, 2] > 0) & (rows[:, 3] > 0))
    rows, classes, scores = rows[mask], classes[mask], scores[mask]
    if not len(rows):
        return np.empty((0, 6), dtype=np.float32)
    boxes = np.column_stack((rows[:, :2] - rows[:, 2:4] / 2,
                             rows[:, :2] + rows[:, 2:4] / 2))
    keep = _class_nms(boxes, scores, classes, nms_iou)
    boxes = boxes[keep] / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return np.column_stack((boxes, scores[keep], classes[keep])).astype(np.float32)


def document_result(detections):
    """Match the earlier detect_document helper, including nulls for none."""
    if not len(detections):
        return {"document_type": "none", "detection_confidence": None, "box_xyxy": None}
    top = detections[int(detections[:, 4].argmax())]
    return {
        "document_type": CLASS_NAMES[int(top[5])],
        "detection_confidence": float(top[4]),
        "box_xyxy": top[:4].tolist(),
    }


class OnnxDocumentDetector:
    """One image per call. ONNX Runtime is imported only when constructing this class."""

    def __init__(self, model_path, providers=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(Path(model_path)),
                                           providers=providers or ["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if CONTRACT_KEY not in metadata:
            raise ValueError("Missing YOLOX contract; export with notebook 10 first.")
        self.contract = json.loads(metadata[CONTRACT_KEY])
        if (self.contract.get("schema") != 1
                or self.contract.get("architecture") != "yolox_tiny"
                or self.contract.get("classes") != list(CLASS_NAMES)
                or self.contract.get("output_format") != "decoded_cxcywh_objectness_class_probabilities"
                or self.contract.get("preprocessing") != "bgr_0_255_top_left_letterbox_114"):
            raise ValueError("Unsupported ONNX document model contract.")
        self.size = int(self.contract["size"])
        self.confidence = float(self.contract["confidence"])
        self.nms_iou = float(self.contract["nms_iou"])
        if (self.size < 32 or self.size % 32 or not 0 <= self.confidence <= 1
                or not 0 <= self.nms_iou <= 1):
            raise ValueError("Invalid model size or thresholds in ONNX contract.")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        expected_rows = sum((self.size // stride) ** 2 for stride in (8, 16, 32))
        if (len(inputs) != 1 or inputs[0].shape != [1, 3, self.size, self.size]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1
                or outputs[0].shape != [1, expected_rows, 7]):
            raise ValueError("ONNX input/output shapes do not match the YOLOX contract.")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def predict_bgr(self, image):
        tensor, ratio = preprocess_bgr(image, self.size)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        height, width = image.shape[:2]
        return decode_predictions(output, ratio, width, height, self.confidence, self.nms_iou)

    def detect_bgr(self, image):
        return document_result(self.predict_bgr(image))

    def detect_document(self, image_path):
        return self.detect_bgr(read_bgr(image_path))
