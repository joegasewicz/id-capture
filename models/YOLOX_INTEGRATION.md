# YOLOX ONNX integration

Run `training/10_yolox_export_onnx.ipynb` after notebook 09. It creates a
candidate, compares ONNX Runtime against PyTorch and the official YOLOX decoder,
then saves the following files only after the checks pass:

- `models/yolox-tiny-retrained.onnx`
- `models/yolox-tiny-retrained.json` (tensor and preprocessing contract)
- `models/yolox-tiny-retrained.parity.json` (sample parity results and model hash)
- `models/yolox-tiny-retrained.LICENSE.txt` (upstream source licence)

The previous `yolo11n-retrained.onnx` remains available. The mobile app's source
is not in this repository, so its loader has not been changed or tested.

## Model contract

| Property | New YOLOX model |
|---|---|
| Input name | `images` |
| Input type/layout | float32 NCHW `[1, 3, 416, 416]` |
| Channel order and range | BGR, 0–255; do not divide by 255 |
| Resize | Preserve aspect ratio: `r = min(416 / height, 416 / width)` |
| Resized dimensions | `int(width * r)`, `int(height * r)`; bilinear interpolation |
| Padding | Copy into the top-left of a 416×416 canvas filled with 114 |
| Output name/layout | `predictions`, float32 `[1, 3549, 7]` |
| Output columns | `cx, cy, width, height, objectness, p_driving_licence, p_passport` |
| Box units | Pixels in the letterboxed input |
| Class IDs | 0 = driving licence; 1 = passport |
| Confidence | `objectness * max(p_driving_licence, p_passport)` |
| Confidence threshold | Read the exported JSON/metadata; currently validation selected 0.20 |
| NMS | Class-aware, IoU threshold 0.65, descending confidence |

Box grid/stride decoding and probability activation are inside this graph.
Do **not** decode grids or apply sigmoid a second time. NMS is outside the graph.
The old YOLO11 model uses `[1, 3, 640, 640]` input and `[1, 6, 8400]` output;
its raw tensor parser is incompatible with this model.

For each prediction, choose the highest-probability class and multiply its
probability by objectness. Keep scores at or above the exported threshold.
Convert `cx,cy,w,h` to `x1,y1,x2,y2`; perform class-aware NMS before clipping.
Divide coordinates by `r`, then clamp x to `[0,width]` and y to `[0,height]`.
Padding is top-left aligned, so no x/y padding subtraction is needed.

Use upright frames: apply file EXIF orientation and camera sensor orientation
before preprocessing. Reverse any preview mirroring when mapping boxes to the
screen; preview cropping/scaling is a separate UI transform. Coordinates refer
to the full upright image, not directly to the preview view.

## Stable application result

The Python reference adapter preserves the earlier notebook's result fields:

```python
from id_capture.onnx_detector import OnnxDocumentDetector

detector = OnnxDocumentDetector("models/yolox-tiny-retrained.onnx")
result = detector.detect_document("capture.jpg")
# {
#     "document_type": "passport",
#     "detection_confidence": 0.94,
#     "box_xyxy": [40.0, 80.0, 420.0, 600.0],
# }
```

No accepted detections returns `document_type="none"`,
`detection_confidence=None`, and `box_xyxy=None`. With multiple accepted boxes,
use the one with the highest confidence. There is no third output class for
`none`, and confidence is not a capture-quality percentage.

`detect_bgr(image)` accepts an upright HWC BGR uint8 frame. `predict_bgr(image)`
returns all post-NMS detections as rows of `x1,y1,x2,y2,confidence,class_id`.
The reference adapter needs ONNX Runtime, NumPy, OpenCV, and Pillow; it does not
load PyTorch, YOLOX source, or Ultralytics.

The same steps must be implemented in the mobile loader before swapping models.
Confirm its ONNX runtime supports opset 12, then check phone latency, camera
orientation, preview-box alignment, target/negative captures, and final result
fields. CPU parity in notebook 10 does not establish mobile compatibility.

## Validation and provenance

Calibration comes from the selected checkpoint's validation results, not test
results. Export checks use validation samples and the existing phone photos,
without retraining. They compare raw geometric/probability tensors and final
detections against the official PyTorch postprocessor. Any mismatch stops
publication. This is numerical equivalence testing, not a new accuracy claim.

Keep the upstream licence and relevant notices with distribution. The export
records the source revision, checkpoint hash, and model hash; review the
pretrained weights and dataset terms as well as the source licence.
