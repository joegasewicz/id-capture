"""Notebook-controlled ONNX export and parity checks; nothing runs on import."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from id_capture.onnx_detector import (
    CLASS_NAMES, CONTRACT_KEY, OnnxDocumentDetector, decode_predictions,
    document_result, preprocess_bgr, read_bgr,
)
from training.yolox_workflow import SOURCE_COMMIT, make_model, save_json, show_progress


def export_candidate(project):
    """Fixed batch-1 CPU graph with box decoding included; NMS stays in the adapter."""
    import onnx

    project = Path(project)
    data = project / "training/data/yolox_all_sources"
    pointer = json.loads((data / "latest_run.json").read_text())
    run = Path(pointer["run"])
    calibration = json.loads((run / "calibration.json").read_text())
    checkpoint = Path(pointer["best"])
    if Path(calibration["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Calibration does not belong to the selected checkpoint.")
    size = int(pointer["size"])
    threshold = float(calibration["threshold"])
    contract = dict(
        schema=1, architecture="yolox_tiny", classes=list(CLASS_NAMES), size=size,
        input_name="images", input_shape=[1, 3, size, size],
        preprocessing="bgr_0_255_top_left_letterbox_114",
        output_name="predictions",
        output_shape=[1, sum((size // s) ** 2 for s in (8, 16, 32)), 7],
        output_format="decoded_cxcywh_objectness_class_probabilities",
        output_columns=["cx", "cy", "width", "height", "objectness", "p_driving_licence", "p_passport"],
        score="objectness * max(class_probabilities)",
        confidence=threshold, nms_iou=0.65, nms="class_aware",
        coordinates="pixels in EXIF-upright original image after inverse letterbox",
        source_commit=SOURCE_COMMIT,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        validation_targets_met=calibration["all_targets_met"],
    )
    final = project / "models/yolox-tiny-retrained.onnx"
    candidate = final.with_name(final.stem + ".candidate.onnx")
    final.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    show_progress("Loading checkpoint", 0, 3, started)
    model = make_model(checkpoint).cpu().float().eval()
    model.head.decode_in_inference = True
    example = torch.zeros(1, 3, size, size)
    with torch.inference_mode():
        # Populate the fixed-size inference grids before tracing.
        model(example)
    show_progress("Exporting ONNX", 1, 3, started)
    with torch.inference_mode():
        torch.onnx.export(
            model, example, str(candidate), input_names=["images"],
            output_names=["predictions"], opset_version=12,
            dynamic_axes=None, dynamo=False,
        )
    graph = onnx.load(candidate)
    metadata = {p.key: p.value for p in graph.metadata_props}
    metadata[CONTRACT_KEY] = json.dumps(contract)
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph)
    onnx.save(graph, candidate)
    show_progress("Candidate exported; parity checks are next", 3, 3, started, finished=True)
    return dict(model=model, candidate=candidate, final=final, contract=contract,
                manifest=Path(pointer["manifest"]), run=run)


def parity_samples(manifest, phone_root, per_group=3):
    """Sample validation by source/class; include all labelled phone photos."""
    records = json.loads(Path(manifest).read_text())["records"]
    samples = []
    for source in sorted({r["source"] for r in records}):
        for cls in (0, 1, None):
            group = [r for r in records if r["source"] == source
                     and r["split"] == "val" and r["class"] == cls]
            for record in group[:per_group]:
                samples.append(dict(path=record["image"], group=f"validation:{source}:{cls}"))
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    for name in (*CLASS_NAMES, "none"):
        folder = Path(phone_root) / name
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in extensions:
                    samples.append(dict(path=str(path), group=f"phone:{name}"))
    if not samples:
        raise ValueError("No validation or phone samples found for parity checks.")
    return samples


def validate_and_publish(bundle, phone_root):
    """Publish only after tensor, postprocessing, and app-result parity pass."""
    from yolox.utils import postprocess

    adapter = OnnxDocumentDetector(bundle["candidate"])
    model = bundle["model"]
    contract = bundle["contract"]
    samples = parity_samples(bundle["manifest"], phone_root)
    rows = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        image = read_bgr(sample["path"])
        tensor, ratio = preprocess_bgr(image, adapter.size)
        height, width = image.shape[:2]
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor)).cpu().numpy()
        actual = adapter.session.run([adapter.output_name], {adapter.input_name: tensor})[0]
        assert actual.shape == tuple(contract["output_shape"])
        # Separate geometric error from probability error; box values are pixels.
        np.testing.assert_allclose(actual[:, :, :4], expected[:, :, :4], rtol=1e-4, atol=0.05)
        np.testing.assert_allclose(actual[:, :, 4:], expected[:, :, 4:], rtol=1e-3, atol=1e-4)
        converted = decode_predictions(actual, ratio, width, height,
                                       adapter.confidence, adapter.nms_iou)
        # Independent reference: official YOLOX Torch postprocessing, not our NumPy decoder.
        reference = postprocess(torch.from_numpy(expected.copy()), 2,
                                adapter.confidence, adapter.nms_iou)[0]
        if reference is None:
            reference = np.empty((0, 6), dtype=np.float32)
        else:
            array = reference.numpy()
            boxes = array[:, :4] / ratio
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
            reference = np.column_stack((boxes, array[:, 4]*array[:, 5], array[:, 6]))
        assert converted.shape == reference.shape, f"Detection count mismatch: {sample['path']}"
        if len(reference):
            # Both implementations return confidence-sorted detections.
            np.testing.assert_array_equal(converted[:, 5], reference[:, 5])
            np.testing.assert_allclose(converted[:, :4], reference[:, :4], rtol=1e-4, atol=0.25)
            np.testing.assert_allclose(converted[:, 4], reference[:, 4], rtol=1e-3, atol=1e-4)
        result = document_result(converted)
        assert result["document_type"] == document_result(reference)["document_type"]
        rows.append(dict(**sample, detections=len(converted), result=result,
                         max_box_tensor_error=float(np.abs(actual[:, :, :4]-expected[:, :, :4]).max()),
                         max_probability_error=float(np.abs(actual[:, :, 4:]-expected[:, :, 4:]).max())))
        show_progress(f"ONNX parity {index}/{len(samples)}", index, len(samples), started)
    # Release the runtime session before renaming (also supports platforms locking files).
    del adapter
    source_license = bundle["final"].parents[1] / "training/vendor/YOLOX/LICENSE"
    if not source_license.is_file():
        raise FileNotFoundError("Keep the upstream YOLOX LICENSE from notebook 07.")
    report = dict(passed=True, images=len(rows), contract=contract, rows=rows,
                  scope="CPU PyTorch/ONNX tensor and detection parity; mobile device not tested",
                  onnx_sha256=hashlib.sha256(bundle["candidate"].read_bytes()).hexdigest())
    # Existing YOLO11 export is never replaced. Candidate is promoted only after all checks.
    final = bundle["final"]
    shutil.copyfile(source_license, final.with_suffix(".LICENSE.txt"))
    bundle["candidate"].replace(final)
    save_json(final.with_suffix(".json"), contract)
    save_json(final.with_suffix(".parity.json"), report)
    show_progress("ONNX parity passed and model published locally", len(rows), len(rows), started, finished=True)
    print("Model:", final)
    print("Contract:", final.with_suffix(".json"))
    print("Parity report:", final.with_suffix(".parity.json"))
    return report
