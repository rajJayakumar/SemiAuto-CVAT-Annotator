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
