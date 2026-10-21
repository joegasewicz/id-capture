"""YOLOX document training helpers. Called by notebooks; no work on import."""

import hashlib
import json
import random
import tarfile
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

NAMES = ["driving_licence", "passport"]
SOURCE_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
PHOTO_DESIGNS = {
    "alb_id": "01_alb_id", "aze_passport": "05_aze_passport",
    "esp_id": "21_esp_id_old", "est_id": "22_est_id",
    "fin_id": "24_fin_id", "grc_passport": "25_grc_passport",
    "lva_passport": "32_lva_passport", "rus_internalpassport": "39_rus_internalpassport",
    "srb_passport": "41_srb_passport", "svk_id": "42_svk_id",
}


def show_progress(label, completed, total, started, finished=False):
    """Local wall-clock finish estimate; unfinished work never shows 100%."""
    elapsed = max(0.0, time.perf_counter() - started)
    fraction = min(1.0, completed / total) if total else 0.0
    percentage = 100.0 if finished else min(99.9, fraction * 100)
    remaining = elapsed / fraction * (1 - fraction) if fraction > 0 else None
    now = datetime.now().astimezone()
    finish = now + timedelta(seconds=remaining or 0)
    eta = str(timedelta(seconds=int(remaining))) if remaining is not None else "estimating"
    finish_text = finish.strftime("%Y-%m-%d %H:%M:%S %Z") if remaining is not None else "estimating"
    if finished:
        eta, finish_text = "0:00:00", now.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{label} | {percentage:.1f}% | Elapsed {timedelta(seconds=int(elapsed))}"
          f" | Remaining {eta} | {'Finished' if finished else 'Estimated finish'} {finish_text}", flush=True)


def document_class(name):
    if "_drvlic" in name:
        return 0
    if "_passport" in name and "passportcard" not in name:
        return 1
    return None


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def clipped_box(quad, width, height):
    quad = np.asarray(quad, dtype=float)
    assert quad.shape == (4, 2) and np.isfinite(quad).all()
    lo = np.maximum(quad.min(axis=0), [0, 0])
    hi = np.minimum(quad.max(axis=0), [width, height])
    return None if np.any(hi <= lo) else np.r_[lo, hi].tolist()


def audit(records):
    """Require design separation across sources, including MIDV-2020."""
    assert records and len({r["image"] for r in records}) == len(records)
    groups = {}
    for split in ("train", "val", "test"):
        part = [r for r in records if r["split"] == split]
        assert {r["class"] for r in part} == {0, 1, None}, split
        groups[split] = {r["design"] for r in part}
        print(split, len(part), dict(Counter(
            (r["source"], "none" if r["class"] is None else NAMES[r["class"]])
            for r in part
        )))
    assert groups["train"].isdisjoint(groups["val"] | groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert all(Path(r["image"]).is_file() for r in records)
    print("PASS: no document design crosses splits; all images exist.")


def prepare_data(project, source_root, output, max_side=960):
    """All video frames + all downloaded 2020 photos; sources remain untouched."""
    project, source_root, output = map(Path, (project, source_root, output))
    assignment = json.loads((project / "training/data/midv500_yolo/manifest.json").read_text())["config"]["assignment"]
    archive_path = source_root / "midv2020/photo.tar"
    assert archive_path.is_file(), archive_path
    jobs = []
    for source in ("midv500", "midv2019"):
        root = source_root / source
        folders = sorted(p for p in root.iterdir() if p.is_dir() and (p / "images").is_dir())
        assert {p.name for p in folders} == set(assignment), source
        for folder in folders:
            for clip in sorted(p for p in (folder / "images").iterdir() if p.is_dir()):
                for image in sorted(clip.iterdir()):
                    if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                        continue
                    annotation = folder / "ground_truth" / clip.name / (image.stem + ".json")
                    assert annotation.is_file(), annotation
                    jobs.append(dict(source=source, design=folder.name,
                                     split=assignment[folder.name], cls=document_class(folder.name),
                                     original=str(image), annotation=str(annotation),
                                     key=f"{source}__{folder.name}__{clip.name}__{image.stem}"))
    with tarfile.open(archive_path, "r:") as archive:
        for code, design in PHOTO_DESIGNS.items():
            with archive.extractfile(f"annotations/{code}.json") as handle:
                annotations = json.load(handle)["_via_img_metadata"]
            assert len(annotations) == 100, code
            for annotation in annotations.values():
                regions = annotation["regions"]
                if isinstance(regions, dict):
                    regions = regions.values()
                shapes = [r["shape_attributes"] for r in regions
                          if r["region_attributes"].get("field_name") == "doc_quad"]
                assert len(shapes) == 1, (code, annotation["filename"])
                shape = shapes[0]
                member = f"images/{code}/{annotation['filename']}"
                assert archive.getmember(member).isfile()
                jobs.append(dict(source="midv2020", design=design, split=assignment[design],
                                 cls=document_class(design), member=member,
                                 quad=np.column_stack((shape["all_points_x"], shape["all_points_y"])).tolist(),
                                 original=f"{archive_path}::{member}",
                                 key=f"midv2020__{code}__{Path(member).stem}"))
    config = dict(schema=1, max_side=max_side, assignment=assignment, jobs=len(jobs),
                  inventory_sha256=hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest())
    print("Available camera images:", dict(Counter(j["source"] for j in jobs)))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["config"] == config, "Inputs changed: choose a new output directory."
        audit(manifest["records"])
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "preparation_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Choose a new output directory."
    else:
        assert not any(output.iterdir()), "Unexpected output directory contents."
        save_json(config_path, config)
    records = []
    started = time.perf_counter()
    with tarfile.open(archive_path, "r:") as archive:
        for index, job in enumerate(jobs, 1):
            if job["source"] == "midv2020":
                with archive.extractfile(job["member"]) as handle:
                    with Image.open(handle) as original:
                        image = np.array(ImageOps.exif_transpose(original).convert("RGB"))[:, :, ::-1].copy()
                quad = job["quad"]
            else:
                image = cv2.imread(job["original"])
                assert image is not None, job["original"]
                quad = json.loads(Path(job["annotation"]).read_text())["quad"]
            height, width = image.shape[:2]
            box = clipped_box(quad, width, height)
            cls = job["cls"] if box is not None else None
            if cls is None:
                box = None
            scale = min(1.0, max_side / max(height, width))
            new_width, new_height = round(width * scale), round(height * scale)
            if scale < 1:
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            if box is not None:
                box = (
                    np.array(box) * [new_width / width, new_height / height, new_width / width, new_height / height]
                ).tolist()
            destination = output / "images" / job["split"] / (job["key"] + ".jpg")
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append(dict(image=str(destination), source=job["source"], design=job["design"],
                                original=job["original"], split=job["split"],
                                **{"class": cls}, box=box, width=new_width, height=new_height))
            if index % 250 == 0 or index == len(jobs):
                show_progress(f"Prepare {index}/{len(jobs)}", index, len(jobs), started)
    audit(records)
    manifest = dict(config=config, records=records)
    save_json(manifest_path, manifest)
    show_progress("Preparation complete", len(jobs), len(jobs), started, finished=True)
    return manifest


def letterbox(image, size):
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    # Official YOLOX preprocessing uses BGR values in 0..255, not 0..1.
    return np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32), ratio


class DocumentDataset(Dataset):
    def __init__(self, records, size=416, augment=False):
        self.records, self.size, self.augment = records, size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(record["image"])
        if image is None:
            raise ValueError(record["image"])
        if self.augment:
            image = np.clip(image.astype(np.float32) * random.uniform(0.75, 1.25)
                            + random.uniform(-15, 15), 0, 255).astype(np.uint8)
        image, ratio = letterbox(image, self.size)
        target = np.zeros((1, 5), dtype=np.float32)
        if record["class"] is not None:
            x1, y1, x2, y2 = np.array(record["box"]) * ratio
            target[0] = [record["class"], (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
        return torch.from_numpy(image), torch.from_numpy(target)


def make_model(checkpoint=None):
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    model = YOLOX(YOLOPAFPN(depth=0.33, width=0.375), YOLOXHead(2, width=0.375))
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eps, layer.momentum = 1e-3, 0.03
    model.head.initialize_biases(0.01)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        current = model.state_dict()
        matching = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        missing = set(current) - set(matching)
        assert all(k.startswith("head.cls_preds.") for k in missing), sorted(missing)
        assert not (set(state) - set(current)), "Unexpected checkpoint architecture"
        model.load_state_dict(matching, strict=False)
        print("Loaded pretrained tensors:", len(matching), "| New class-head tensors:", len(missing))
    return model


def iou(boxes, target):
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], target[2:]) - np.maximum(boxes[:, :2], target[:2])).prod(axis=1)
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    area = np.maximum(0, target[2:] - target[:2]).prod()
    return intersection / np.maximum(areas + area - intersection, 1e-9)


@torch.inference_mode()
def predict_records(model, records, size=416, batch=16, minimum_confidence=0.05, progress=None):
    from yolox.utils import postprocess

    model.eval()
    device = next(model.parameters()).device
    predictions = []
    started = time.perf_counter()
    for offset in range(0, len(records), batch):
        part = records[offset:offset+batch]
        prepared = [letterbox(cv2.imread(r["image"]), size) for r in part]
        images = torch.from_numpy(np.stack([p[0] for p in prepared])).to(device)
        outputs = postprocess(model(images), 2, minimum_confidence, 0.65)
        for output, (_, ratio), record in zip(outputs, prepared, part):
            if output is None:
                predictions.append(np.empty((0, 6), dtype=np.float32))
            else:
                array = output.cpu().numpy()
                boxes = array[:, :4] / ratio
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, record["width"])
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, record["height"])
                predictions.append(np.column_stack((boxes, array[:, 4] * array[:, 5], array[:, 6])))
        done = min(offset + batch, len(records))
        if offset % (batch * 50) == 0 or done == len(records):
            if progress is None:
                show_progress(f"Inference {done}/{len(records)}", done, len(records), started,
                              finished=done == len(records))
            else:
                progress(done / len(records))
    return predictions


def summarize(records, predictions, threshold):
    assert len(records) == len(predictions)
    rows = []
    for source in ["combined"] + sorted({r["source"] for r in records}):
        totals, hits = Counter(), Counter()
        false_positives = 0
        for record, prediction in zip(records, predictions):
            if source != "combined" and record["source"] != source:
                continue
            cls = record["class"]
            totals[cls] += 1
            selected = prediction[prediction[:, 4] >= threshold]
            if cls is None:
                false_positives += bool(len(selected))
            else:
                correct = selected[selected[:, 5] == cls]
                hits[cls] += bool(len(correct) and np.any(iou(correct[:, :4], np.array(record["box"])) >= 0.5))
        rows.append(dict(source=source, threshold=float(threshold), images=sum(totals.values()),
                         licence_count=totals[0], passport_count=totals[1], negative_count=totals[None],
                         licence_recall=hits[0]/totals[0] if totals[0] else None,
                         passport_recall=hits[1]/totals[1] if totals[1] else None,
                         negative_fp=false_positives/totals[None] if totals[None] else None))
    return rows


def train_model(manifest, checkpoint, runs, epochs=40, batch=16, size=416, seed=42):
    """Notebook-native single-GPU fine-tuning; test records are never loaded."""
    assert torch.cuda.is_available(), "CUDA is required."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    records = manifest["records"]
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "val"]
    audit(records)
    run = Path(runs) / ("yolox_tiny_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run.mkdir(parents=True)
    config = dict(epochs=epochs, batch=batch, size=size, seed=seed, initial_checkpoint=str(checkpoint),
                  source_commit=SOURCE_COMMIT, dataset=manifest["config"],
                  selection="validation mean class recall minus negative false-positive rate at confidence 0.25, IoU 0.5",
                  augmentation="brightness/contrast only; no mosaic, mixup, or flips")
    save_json(run / "config.json", config)
    model = make_model(checkpoint).cuda()
    loader = DataLoader(DocumentDataset(train, size, augment=True), batch_size=batch,
                        shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00001)
    scaler = torch.amp.GradScaler("cuda")
    best_score = -float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, (images, targets) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                losses = model(images.cuda(non_blocking=True), targets.cuda(non_blocking=True))
                loss = losses["total_loss"]
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}, batch {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if step % 50 == 0 or step == len(loader):
                # Reserve part of each epoch for validation and checkpoint writes.
                completed = epoch + 0.90 * step / len(loader)
                show_progress(f"Epoch {epoch+1}/{epochs}, batch {step}/{len(loader)}, loss {running/step:.4f}",
                              completed, epochs, started)
        scheduler.step()
        def validation_progress(fraction):
            show_progress(f"Epoch {epoch+1}/{epochs}, validation {fraction:.1%}",
                          epoch + 0.90 + 0.09 * fraction, epochs, started)
        predictions = predict_records(model, validation, size, batch, 0.25, validation_progress)
        metrics = summarize(validation, predictions, 0.25)[0]
        score = (metrics["licence_recall"] + metrics["passport_recall"]) / 2 - metrics["negative_fp"]
        history.append(dict(epoch=epoch+1, loss=running/len(loader), score=score, **metrics))
        print("Validation:", history[-1], flush=True)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     epoch=epoch+1, config=config)
        torch.save(state, run / "last.pth")
        if score > best_score:
            best_score = score
            torch.save(state, run / "best.pth")
        save_json(run / "history.json", history)
        show_progress(f"Saved epoch {epoch+1}/{epochs}", epoch+1, epochs, started)
    save_json(run / "completed.json", dict(best=str(run / "best.pth"), elapsed_seconds=time.perf_counter()-started))
    show_progress("Training complete", epochs, epochs, started, finished=True)
    print("Finished. Best checkpoint:", run / "best.pth")
    return run
