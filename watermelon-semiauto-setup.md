# Watermelon CVAT Pre-Annotation Setup

You are setting up a Python project that runs YOLOv8 segmentation inference on watermelon images locally, then uploads polygon pre-annotations to a CVAT task hosted on `app.cvat.ai`. This document is your complete spec — execute all phases in order.

## Project context

- **Goal**: Pre-annotate watermelon images in CVAT so annotators only have to touch up polygons rather than draw from scratch.
- **Classes**: `watermelon` (whole melon outline), `flesh_left` (left half of inner flesh, no rind), `flesh_right` (right half of inner flesh, no rind).
- **Model**: Fine-tuned YOLOv8 segmentation model at `./best.pt`.
- **CVAT host**: `https://app.cvat.ai` (hosted, free tier — Nuclio NOT available, must use SDK upload).

## What the user is providing

The user will place these in the project root before running you:

```
.
├── best.pt              # fine-tuned YOLOv8-seg weights
├── .env                 # CVAT credentials (see Phase 1)
└── images/              # folder of .jpg watermelon images
    ├── melon_001.jpg
    ├── melon_002.jpg
    └── ...
```

**Do not overwrite any of these.** If `best.pt`, `.env`, or `images/` are missing, stop and tell the user what's missing before doing anything else.

## What you will create

```
.
├── requirements.txt
├── verify_model.py      # Phase 2 — sanity check best.pt
├── create_task.py       # Phase 3 — creates the CVAT task and uploads images
├── preannotate.py       # Phase 4 — runs inference and uploads polygons
├── README.md            # how the user runs everything
└── .gitignore
```

---

## Phase 1 — Validate inputs

Before generating any files, run these checks. If any fail, stop and report to the user.

1. Confirm `./best.pt` exists and is a non-empty file.
2. Confirm `./.env` exists and contains keys `CVAT_HOST`, `CVAT_USERNAME`, `CVAT_PASSWORD`. If `CVAT_HOST` is missing, tell the user to add `CVAT_HOST=https://app.cvat.ai`.
3. Confirm `./images/` exists and contains at least one `.jpg`, `.jpeg`, or `.png` file. Note the count for later.
4. Confirm Python 3.10+ is available (`python --version` or `python3 --version`).

---

## Phase 2 — Generate environment and verification

### Create `requirements.txt`

```
ultralytics>=8.3.0
cvat-sdk>=2.16.0
opencv-python>=4.9.0
python-dotenv>=1.0.0
numpy>=1.24.0
Pillow>=10.0.0
```

### Create `.gitignore`

```
.venv/
__pycache__/
*.pyc
.env
best.pt
images/
runs/
.DS_Store
```

### Create a virtual environment and install

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails on `opencv-python` (common on minimal Linux), install system dep first: `apt-get install -y libgl1` (or tell the user to do this if you lack sudo).

### Create `verify_model.py`

```python
"""Verify best.pt is a segmentation model and print its class map."""
from ultralytics import YOLO


def main():
    model = YOLO("best.pt")
    print(f"Task: {model.task}")
    print(f"Classes: {model.names}")

    if model.task != "segment":
        raise SystemExit(
            f"ERROR: Model task is '{model.task}', expected 'segment'. "
            "This script only works with YOLOv8-seg models. "
            "Retrain using a *-seg.pt base (e.g. yolov8n-seg.pt)."
        )

    expected = {"watermelon", "flesh_left", "flesh_right"}
    actual = set(model.names.values())
    missing = expected - actual
    extra = actual - expected

    if missing:
        print(f"\nWARNING: Model missing expected classes: {missing}")
        print(f"Model has: {actual}")
        print("You will need to update CLASS_REMAP in preannotate.py")
    if extra:
        print(f"\nNOTE: Model has extra classes not in target set: {extra}")
        print("These will be skipped during upload.")

    print("\nModel verification complete.")


if __name__ == "__main__":
    main()
```

### Run verification

Execute `python verify_model.py`. Capture the output. If `model.task != "segment"`, STOP and tell the user to retrain — there is no point continuing.

Save the printed `Classes:` dict (the `model.names` mapping). You will need it for Phase 3 and Phase 4.

---

## Phase 3 — Create CVAT task and upload images

### Create `create_task.py`

```python
"""Create a CVAT task with the right labels and upload all images from ./images/."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType

load_dotenv()

TASK_NAME = os.environ.get("TASK_NAME", "watermelon_preannotated_batch_1")
IMAGE_DIR = Path("images")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_images() -> list[str]:
    paths = sorted(
        str(p) for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        sys.exit(f"No images found in {IMAGE_DIR}/")
    return paths


def main():
    images = collect_images()
    print(f"Found {len(images)} images")

    with make_client(
        host=os.environ["CVAT_HOST"],
        credentials=(os.environ["CVAT_USERNAME"], os.environ["CVAT_PASSWORD"]),
    ) as client:
        task_spec = {
            "name": TASK_NAME,
            "labels": [
                {"name": "watermelon",  "color": "#33ddff", "type": "polygon"},
                {"name": "flesh_left",  "color": "#ff6633", "type": "polygon"},
                {"name": "flesh_right", "color": "#ffcc33", "type": "polygon"},
            ],
        }

        print(f"Creating task '{TASK_NAME}'...")
        task = client.tasks.create_from_data(
            spec=task_spec,
            resource_type=ResourceType.LOCAL,
            resources=images,
        )

    print(f"\n{'=' * 50}")
    print(f"TASK CREATED — ID: {task.id}")
    print(f"{'=' * 50}")
    print(f"\nNext step: set TASK_ID={task.id} in your .env file, then run:")
    print(f"  python preannotate.py")


if __name__ == "__main__":
    main()
```

### Run task creation

Execute `python create_task.py`. Watch for:

- **`401 Unauthorized`** → credentials in `.env` are wrong. App.cvat.ai locks accounts after ~4 bad attempts for ~10 minutes. Stop and ask user to verify credentials.
- **Long upload time** → normal. Free tier throttles uploads. 100 images can take 5-10 minutes.
- **Successful run** → prints `TASK CREATED — ID: <number>`. **Append `TASK_ID=<number>` to `.env`** automatically (use the `python-dotenv` approach below) and tell the user.

Update `.env` programmatically:
```python
from dotenv import set_key
set_key(".env", "TASK_ID", str(task.id))
```
Or, simpler, append a line: `echo "TASK_ID=<id>" >> .env`.

---

## Phase 4 — Pre-annotation script

### Create `preannotate.py`

```python
"""Run YOLOv8-seg inference on local images and upload polygons to CVAT task."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from cvat_sdk import make_client

load_dotenv()

# --- Config (override via .env if desired) ---
TASK_ID = int(os.environ["TASK_ID"])
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "images"))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))
MAX_POLY_VERTICES = int(os.environ.get("MAX_POLY_VERTICES", "150"))
APPROX_EPSILON = float(os.environ.get("APPROX_EPSILON", "1.5"))
ENFORCE_LEFT_RIGHT_BY_CENTROID = os.environ.get("ENFORCE_LR", "1") == "1"

# If your model's class names differ from CVAT label names, remap here.
# Key = model class name, Value = CVAT label name.
CLASS_REMAP = {
    "watermelon": "watermelon",
    "flesh_left": "flesh_left",
    "flesh_right": "flesh_right",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def simplify_polygon(points: np.ndarray) -> np.ndarray:
    """Reduce vertex count using cv2.approxPolyDP, fall back to uniform downsample."""
    if len(points) <= MAX_POLY_VERTICES:
        return points
    pts_int = points.astype(np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts_int, APPROX_EPSILON, closed=True)
    simplified = approx.reshape(-1, 2).astype(float)
    if len(simplified) < 10:
        idx = np.linspace(0, len(points) - 1, MAX_POLY_VERTICES).astype(int)
        simplified = points[idx]
    return simplified


def run():
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print(f"Model classes: {model.names}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")

    with make_client(
        host=os.environ["CVAT_HOST"],
        credentials=(os.environ["CVAT_USERNAME"], os.environ["CVAT_PASSWORD"]),
    ) as client:
        task = client.tasks.retrieve(TASK_ID)
        print(f"Retrieved task #{TASK_ID}: '{task.name}'")

        # Build CVAT label name -> id map
        cvat_labels = {lbl.name: lbl.id for lbl in task.get_labels()}
        print(f"CVAT labels: {cvat_labels}")

        # Validate every model class can be mapped to a CVAT label
        for model_cls in model.names.values():
            cvat_name = CLASS_REMAP.get(model_cls)
            if cvat_name is None:
                print(f"  [skip] model class '{model_cls}' has no remap entry — will be ignored")
                continue
            if cvat_name not in cvat_labels:
                sys.exit(
                    f"ERROR: CVAT task has no label '{cvat_name}' "
                    f"(mapped from model class '{model_cls}'). "
                    f"Available: {list(cvat_labels.keys())}"
                )

        # Map frame filename -> CVAT frame index
        frames_info = task.get_frames_info()
        frame_index = {Path(f.name).name: i for i, f in enumerate(frames_info)}
        print(f"Task has {len(frame_index)} frames\n")

        shapes = []
        stats = {"processed": 0, "skipped": 0, "polygons": 0, "swapped_lr": 0}

        image_paths = sorted(
            p for p in IMAGE_DIR.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )

        for img_path in image_paths:
            if img_path.name not in frame_index:
                stats["skipped"] += 1
                continue
            frame = frame_index[img_path.name]

            results = model(
                str(img_path),
                conf=CONF_THRESHOLD,
                retina_masks=True,
                verbose=False,
            )

            frame_detections = []
            for r in results:
                if r.masks is None:
                    continue
                for poly_xy, cls_idx, conf in zip(
                    r.masks.xy,
                    r.boxes.cls.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                ):
                    model_cls = model.names[int(cls_idx)]
                    cvat_cls = CLASS_REMAP.get(model_cls)
                    if cvat_cls is None:
                        continue

                    pts = simplify_polygon(poly_xy.astype(float))
                    if len(pts) < 3:
                        continue

                    frame_detections.append({
                        "cvat_cls": cvat_cls,
                        "points": pts,
                        "centroid_x": float(pts[:, 0].mean()),
                        "conf": float(conf),
                    })

            # Enforce left/right by centroid x-position
            if ENFORCE_LEFT_RIGHT_BY_CENTROID:
                left = [d for d in frame_detections if d["cvat_cls"] == "flesh_left"]
                right = [d for d in frame_detections if d["cvat_cls"] == "flesh_right"]
                if len(left) == 1 and len(right) == 1:
                    if left[0]["centroid_x"] > right[0]["centroid_x"]:
                        left[0]["cvat_cls"] = "flesh_right"
                        right[0]["cvat_cls"] = "flesh_left"
                        stats["swapped_lr"] += 1

            for det in frame_detections:
                shapes.append({
                    "type": "polygon",
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "rotation": 0,
                    "points": det["points"].flatten().tolist(),
                    "frame": frame,
                    "label_id": cvat_labels[det["cvat_cls"]],
                    "group": 0,
                    "source": "auto",
                    "attributes": [],
                })

            stats["processed"] += 1
            stats["polygons"] += len(frame_detections)
            print(f"[{img_path.name}] frame {frame}: {len(frame_detections)} polygons")

        print(f"\n--- Summary ---")
        print(f"Processed: {stats['processed']}")
        print(f"Skipped (not in task): {stats['skipped']}")
        print(f"Total polygons: {stats['polygons']}")
        print(f"Left/right swaps applied: {stats['swapped_lr']}")

        if not shapes:
            sys.exit("ERROR: No polygons produced. Try lowering CONF_THRESHOLD.")

        print(f"\nUploading {len(shapes)} polygons to task {TASK_ID}...")
        task.set_annotations({
            "version": 0,
            "tags": [],
            "shapes": shapes,
            "tracks": [],
        })
        print("Done. Open the task in CVAT to verify.")


if __name__ == "__main__":
    run()
```

### Important behavioral notes for this script

- `retina_masks=True` is **required** — without it polygon coordinates are in the model's input resolution (640×640) instead of the original image, and they will appear shifted/scaled wrong in CVAT.
- `task.set_annotations(...)` **overwrites** all annotations on the task. If you re-run this after an annotator has touched up polygons, their work is lost. The script should print a confirmation prompt before uploading if it detects that the task already has annotations — but for the first run, just upload.
- If the user's model class names differ from `watermelon` / `flesh_left` / `flesh_right`, update `CLASS_REMAP` in `preannotate.py` based on what `verify_model.py` printed in Phase 2. Show the user the file and tell them to verify.

---

## Phase 5 — Create README.md for the user

```markdown
# Watermelon CVAT Pre-Annotation

Local YOLOv8-seg inference → polygon upload to app.cvat.ai.

## Setup (one-time)

1. Place `best.pt`, populated `.env`, and `images/` folder in this directory.
2. `.env` must contain:
   ```
   CVAT_HOST=https://app.cvat.ai
   CVAT_USERNAME=your_username
   CVAT_PASSWORD=your_password
   ```
3. Activate the venv: `source .venv/bin/activate`

## Workflow

```bash
# 1. Verify model is a segmentation model with expected classes
python verify_model.py

# 2. Create the CVAT task and upload images (writes TASK_ID into .env)
python create_task.py

# 3. Run inference and upload pre-annotations
python preannotate.py
```

Open the task at https://app.cvat.ai → your task → Job #1 to verify and touch up.

## Tuning

Override defaults via .env:
- `CONF_THRESHOLD` (default 0.35) — lower = more polygons, more touch-up needed
- `MAX_POLY_VERTICES` (default 150) — lower if CVAT UI lags
- `ENFORCE_LR` (default 1) — set to 0 to disable left/right centroid swap

## Re-running

`preannotate.py` OVERWRITES existing annotations on the task. Do not re-run after annotators start editing — create a new task instead.

For a new batch:
1. Replace contents of `images/`
2. Either reuse the existing task (clear `TASK_ID` and re-run `create_task.py`)
3. Or, more commonly, create a fresh task for each batch
```

---

## Phase 6 — Final handoff

After all files are created and the environment is set up, do this:

1. Run `python verify_model.py` and show the user the output.
2. Tell the user:
   - "Setup complete. To proceed, run `python create_task.py` to create the CVAT task and upload images. This may take several minutes. When it finishes, run `python preannotate.py`."
   - If the model's class names differ from the expected `watermelon` / `flesh_left` / `flesh_right`, point this out explicitly and tell them to update `CLASS_REMAP` in `preannotate.py`.
3. Do NOT run `create_task.py` or `preannotate.py` yourself unless the user asks. Those are user-initiated because they touch their CVAT account.

## Troubleshooting reference (include in your final message to the user)

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` from CVAT | Bad credentials in `.env` | Verify username/password. Wait 10 min if locked. |
| Polygons appear shifted in CVAT | `retina_masks=True` missing | Already set in the script — don't remove it. |
| Polygons load slowly in CVAT UI | Too many vertices | Lower `MAX_POLY_VERTICES` or raise `APPROX_EPSILON` |
| No polygons produced | Threshold too high, or wrong model | Lower `CONF_THRESHOLD` to 0.25; verify with `model.predict(img, save=True)` |
| Upload succeeds, no polygons visible | Filename mismatch between local and CVAT | Print `list(frame_index.keys())[:5]` and compare to local filenames |
| `KeyError` on label lookup | Class names mismatch | Update `CLASS_REMAP` in `preannotate.py` |
| Model task is `detect` not `segment` | Wrong base model used in training | Retrain with `yolov8n-seg.pt` (or larger seg variant) |