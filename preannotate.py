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

TASK_ID = int(os.environ["TASK_ID"])
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "images"))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))
MAX_POLY_VERTICES = int(os.environ.get("MAX_POLY_VERTICES", "150"))
APPROX_EPSILON = float(os.environ.get("APPROX_EPSILON", "1.5"))
ENFORCE_LEFT_RIGHT_BY_CENTROID = os.environ.get("ENFORCE_LR", "1") == "1"

# Model class name -> CVAT label name.
# "watermelon_whole" in the model maps to "watermelon" label in CVAT.
CLASS_REMAP = {
    "watermelon_whole": "watermelon_whole",
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
        client.organization_slug = os.environ.get("CVAT_ORG", "")
        task = client.tasks.retrieve(TASK_ID)
        print(f"Retrieved task #{TASK_ID}: '{task.name}'")

        cvat_labels = {lbl.name: lbl.id for lbl in task.get_labels()}
        print(f"CVAT labels: {cvat_labels}")

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
