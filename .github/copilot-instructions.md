ok # Copilot Instructions — yolo-license-detector

## Strict rule: read-only assistant
- The ONLY files the assistant may create or edit in this repo are:
  - `.github/copilot-instructions.md`
  - `AGENTS.md`
- Even those two files may only be changed when the developer explicitly asks.
- NEVER create, edit, rename, or delete any other file in this repository.
- NEVER run commands that modify the repo or environment (e.g. `git commit`, `git push`, `git add`, `git reset`, `poetry add`, `pip install`, file writes/redirects).
- NEVER commit or push code.
- ONLY help by giving the developer hints, explanations, and code snippets in the chat window.
- The developer writes and applies all code changes themselves.
- Reading files to understand context is allowed.
- If asked to make a change, reply with a suggested snippet and where it should go, and do not apply it.

## Project goal
Build a **library** and a **mobile app** for ID document capture.

The user holds a **driving licence card** or a **passport** up to a mobile phone camera, and the system returns:
1. **Document type**: `driving_licence`, `passport`, or `none` (no valid ID detected), with a confidence score.
2. **Capture quality**: a percentage (0–100%) for how good the capture is. Possible inputs include sharpness/blur, glare, lighting/exposure, framing (whole document in view), perspective/skew, and resolution.
3. **Save and upload**: save the captured image (cropped/deskewed document) to disk/storage and upload it.

## Technical direction (open — not locked to YOLO)
- The model/approach is not fixed. YOLO is one option, and better alternatives are welcome, e.g.:
  - Lightweight classifiers (MobileNetV3, EfficientNet-Lite) for document type.
  - Detection/segmentation models (YOLO11-seg, etc.) for locating the document and its corners.
  - On-device SDKs (e.g. Google ML Kit, Apple Vision) for document edge detection.
- The model must be able to run on mobile devices (export to TFLite / Core ML / ONNX).
- Quality scoring can combine classical CV (e.g. Laplacian variance for blur, highlight detection for glare) with model confidence.
- Candidate public datasets: MIDV-500, MIDV-2019, MIDV-2020, plus custom collected samples.
- Python >= 3.13, managed with Poetry.
