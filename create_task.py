"""Create a CVAT task with the right labels and upload all images from ./images/."""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv, set_key
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType

load_dotenv()

TASK_NAME = os.environ.get("TASK_NAME", "watermelon_preannotated_batch_1")
IMAGE_DIR = Path("images")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BATCH_SIZE = 10
MAX_RETRIES = 3


def collect_images() -> list[str]:
    paths = sorted(
        str(p) for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        sys.exit(f"No images found in {IMAGE_DIR}/")
    return paths


def make_cvat_client():
    client = make_client(
        host=os.environ["CVAT_HOST"],
        credentials=(os.environ["CVAT_USERNAME"], os.environ["CVAT_PASSWORD"]),
    )
    client.api_client.rest_client.pool_manager.connection_pool_kw["timeout"] = 600
    return client


def main():
    images = collect_images()
    print(f"Found {len(images)} images")

    with make_cvat_client() as client:
        task_spec = {
            "name": TASK_NAME,
            "labels": [
                {"name": "watermelon",  "color": "#33ddff", "type": "polygon"},
                {"name": "flesh_left",  "color": "#ff6633", "type": "polygon"},
                {"name": "flesh_right", "color": "#ffcc33", "type": "polygon"},
            ],
        }

        # Create the task first (no images yet)
        print(f"Creating task '{TASK_NAME}'...")
        task = client.tasks.create_from_data(
            spec=task_spec,
            resource_type=ResourceType.LOCAL,
            resources=images[:BATCH_SIZE],
        )
        print(f"Task created (ID: {task.id}), uploaded first {min(BATCH_SIZE, len(images))} images")

        # Upload remaining images in batches
        remaining = images[BATCH_SIZE:]
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 2
            total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} images)...")
                    task.upload_data(
                        resource_type=ResourceType.LOCAL,
                        resources=batch,
                    )
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        wait = 10 * attempt
                        print(f"  Batch {batch_num} failed ({e}), retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"\nERROR: Batch {batch_num} failed after {MAX_RETRIES} attempts.")
                        print(f"Task was partially created with ID: {task.id}")
                        print("You can delete it in CVAT and retry, or reduce BATCH_SIZE.")
                        raise

    print(f"\n{'=' * 50}")
    print(f"TASK CREATED — ID: {task.id}")
    print(f"All {len(images)} images uploaded successfully.")
    print(f"{'=' * 50}")

    set_key(".env", "TASK_ID", str(task.id))
    print(f"\nTASK_ID={task.id} has been written to .env")
    print(f"\nNext step: run  python preannotate.py")


if __name__ == "__main__":
    main()
