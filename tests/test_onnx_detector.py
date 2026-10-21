"""Deployment contract checks, independent of trained weights and notebooks."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from id_capture.onnx_detector import (
    CONTRACT_KEY, OnnxDocumentDetector, decode_predictions, document_result,
    preprocess_bgr, read_bgr,
)


class DecoderTests(unittest.TestCase):
    def test_bgr_pixels_top_left_padding_and_resize_ratio(self):
        image = np.full((100, 200, 3), [10, 20, 250], dtype=np.uint8)
        tensor, ratio = preprocess_bgr(image, 416)
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(ratio, 2.08)
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [10, 20, 250])
        np.testing.assert_array_equal(tensor[0, :, 208, 0], [114, 114, 114])

    def test_nms_is_class_aware_and_scores_multiply_objectness(self):
        # Same location: lower-score duplicate licence is suppressed, passport survives.
        raw = np.array([[
            [100, 100, 40, 20, .8, .9, .1],
            [100, 100, 40, 20, .7, .9, .1],
            [100, 100, 40, 20, .9, .1, .9],
            [100, 100, 40, 20, .1, .9, .1],
        ]], dtype=np.float32)
        detections = decode_predictions(raw, 2, 200, 100)
        self.assertEqual(len(detections), 2)
        np.testing.assert_array_equal(detections[:, 5], [1, 0])
        np.testing.assert_allclose(detections[:, 4], [.81, .72])
        np.testing.assert_allclose(detections[0, :4], [40, 45, 60, 55])
        self.assertEqual(document_result(detections)["document_type"], "passport")

    def test_none_keeps_previous_public_null_fields(self):
        detections = decode_predictions(np.zeros((1, 3, 7), dtype=np.float32), 1, 100, 100)
        self.assertEqual(document_result(detections), {
            "document_type": "none", "detection_confidence": None, "box_xyxy": None,
        })

    def test_clips_in_original_image_coordinates(self):
        raw = np.array([[[20, 20, 80, 80, .9, .9, .1]]], dtype=np.float32)
        detection = decode_predictions(raw, 2, 25, 20)[0]
        np.testing.assert_array_equal(detection[:4], [0, 0, 25, 20])

    def test_invalid_layout_and_rgb_float_input_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_predictions(np.zeros((1, 6, 8400)), 1, 100, 100)
        with self.assertRaises(ValueError):
            preprocess_bgr(np.zeros((10, 10, 3), dtype=np.float32), 416)

    def test_file_reader_applies_exif_orientation(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (20, 10), (250, 20, 10))
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(read_bgr(path).shape, (20, 10, 3))


@unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                     "ONNX deployment packages are not installed")
class RuntimeTests(unittest.TestCase):
    def test_embedded_contract_runtime_and_wrong_model_rejection(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        rows = np.zeros((1, 3549, 7), dtype=np.float32)
        rows[0, 0] = [208, 104, 208, 104, .9, .8, .1]
        graph = helper.make_graph(
            [helper.make_node("Constant", [], ["predictions"], value=numpy_helper.from_array(rows))],
            "test_document_contract",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 416, 416])],
            [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 3549, 7])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=8)
        contract = dict(schema=1, architecture="yolox_tiny", classes=["driving_licence", "passport"],
                        size=416, confidence=.2, nms_iou=.65,
                        preprocessing="bgr_0_255_top_left_letterbox_114",
                        output_format="decoded_cxcywh_objectness_class_probabilities")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            onnx.save(model, path)
            with self.assertRaisesRegex(ValueError, "Missing YOLOX contract"):
                OnnxDocumentDetector(path)
            helper.set_model_props(model, {CONTRACT_KEY: json.dumps(contract)})
            onnx.save(model, path)
            detector = OnnxDocumentDetector(path)
            result = detector.detect_bgr(np.zeros((100, 200, 3), dtype=np.uint8))
            self.assertEqual(result["document_type"], "driving_licence")
            self.assertAlmostEqual(result["detection_confidence"], .72, places=5)
            np.testing.assert_allclose(result["box_xyxy"], [50, 25, 150, 75])


if __name__ == "__main__":
    unittest.main()
