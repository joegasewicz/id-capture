# AGENTS.md — examples

> Project guidance and agreed plan, updated 2026-10-09.

## Strict rule: read-only assistant
- The assistant MUST NOT update or change code itself in any way, directly or indirectly. This includes creating, editing, deleting, renaming, formatting, refactoring, generating, or applying patches to code, and using tools, commands, scripts, or other agents to make those changes.
- Requests to implement or fix code must be answered only with explanations and suggested code snippets in chat. The developer must apply every code change themselves.
- The ONLY files the assistant may create or edit in this repo are:
  - `.github/copilot-instructions.md`
  - `AGENTS.md`
- Even those files may only be changed when the developer explicitly asks.
- NEVER create, edit, rename, or delete any other file in this repository.
- Outside explicitly requested edits to the two permitted instruction files, NEVER run commands that modify the repo or environment (e.g. `git commit`, `git push`, `git add`, `git reset`, `poetry add`, `pip install`, file writes/redirects).
- NEVER commit or push code.
- ONLY help by giving the developer hints, explanations, and code snippets in the chat window.
- The developer writes and applies all code changes themselves.
- Reading files to understand context is allowed.
- If asked to change application code, reply with a suggested snippet and where it should go, and do not apply it. Explicitly requested edits to the permitted instruction files may be applied.

## Project goal
Build a **library** and a **mobile app** for ID document capture.

The user holds a **photo driving licence card** or a **passport** up to a mobile phone camera, and the system returns:
1. **Document type**: `driving_licence`, `passport`, or `none` (no valid ID detected), with a confidence score.
2. **Capture quality**: a percentage (0–100%) for how good the capture is. Possible inputs include sharpness/blur, glare, lighting/exposure, framing (whole document in view), perspective/skew, and resolution.
3. **Save and upload**: save the captured image (cropped/deskewed document) to disk/storage and upload it.

The scope includes multiple countries, not only UK documents. Recognition of document type does not establish authenticity or legal validity.

## Agreed learning and model plan
The current OpenCV contour-based approach has proved brittle and too slow for the intended use. Move primary document recognition and localisation to learned models.

1. **Build our own small model from scratch in PyTorch as a learning exercise.** Initially assume at most one target document per image. Predict document type (`passport`, `driving_licence`, or `none`) and a bounding box for a detected target. Learn the architecture, dataset loading, losses, training loop, and evaluation. Handle `none` without requiring a target bounding box.
2. **Fine-tune an existing pretrained detector.** Use the same dataset splits and evaluation criteria to compare against our custom model. YOLO is a candidate; the exact architecture and Hugging Face checkpoint have not yet been selected. Checkpoint hosting and training compute are separate choices.

An optional intermediate experiment is to train that existing architecture from random weights, then compare it with its pretrained version. This is separate from building our own network and is not required for the agreed two-stage path.

- Start with document type and bounding boxes. Add four-corner prediction for cropping and perspective correction later, then capture-quality scoring and save/upload integration.
- Compare detection accuracy, missed targets, false positives on non-target images, training time, and inference latency.
- The eventual production model must support mobile deployment; evaluate export options such as TFLite, Core ML, or ONNX for the selected model.
- OpenCV can remain useful for resizing, perspective correction, and classical quality measurements after detection.
- Keep quality percentage separate from document-class confidence; quality scoring needs its own definition and validation.

## Downloaded training data
- Location: `/home/joe/training-data/midv500` on Ubuntu's internal drive. Use this location rather than the earlier USB or `/Volumes/Joe/...` paths.
- MIDV-500 download completed on 2026-10-09: all **50 archives** passed ZIP verification and were extracted. Original ZIPs are retained.
- Verified **15,000 frame images** and **15,000 matching frame annotation files**, with no missing matching images.
- Frame counts under the initial grouping: **3,600 driving-licence frames**, **4,200 passport frames**, and **7,200 other-document frames**. Reference images are additional to these frame counts.
- Each document folder contains `images`, `ground_truth`, and `videos`. Frame JSON annotations contain a `quad` with four document corners; labels still need conversion for the selected training approach.
- Archives and extracted files occupy approximately **78 GB** together.
- The dataset contains no UK passport or UK driving-licence types. Add suitable examples later if UK coverage is required.
- Inspect the actual documents before assigning final classes. Internal passports, passport cards, and other ID types should not automatically be treated as the target passport class because their filenames contain `passport`.
- Other-document images can support rejection training, but also collect ordinary backgrounds and confusing non-ID objects to evaluate `none`.

## Evaluation limitations
- The 15,000 frames represent only **50 document specimens**, not 15,000 independent documents. Adjacent frames are highly similar.
- Do not randomly mix frames from the same capture sequence across training, validation, and test sets.
- Hold out entire document specimens or types where feasible to assess generalisation beyond documents seen during training. Keep the evaluation split fixed when comparing models.
- This dataset is enough for initial learning experiments; it does not establish production reliability across countries, document versions, cameras, or capture conditions.

## Resume next session
- Downloads are complete. **No training, label conversion, or custom model implementation has started.**
- The user wants to discuss the next step before any further implementation or training. Do not automatically start training or additional downloads.
- The agreed direction is our own small model first, then a pretrained detector. The optional existing-architecture-from-scratch experiment can be skipped.
- When the user resumes, discuss dataset preparation, evaluation splits, and available training hardware, then provide code and steps within the read-only rules.
- Python is managed with Poetry. The active environment in this session was Python **3.12**; `examples/pyproject.toml` currently specifies `>=3.12,<3.14`. Consult the Python environment tools before running any Python toolchain commands.

## This sub-project: `examples/`
- A separate Poetry project (`name = "examples"`) with `tornado` and `marshmallow` as dependencies.
- Its purpose is demo/example usage of the `id-capture` library (e.g. an upload API).
- Entry point: `main.py`.
