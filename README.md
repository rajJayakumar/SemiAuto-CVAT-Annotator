# Watermelon CVAT Pre-Annotation

Local YOLOv8-seg inference → polygon upload to app.cvat.ai.

## Setup (one-time)

1. Place `best.pt`, populated `.env`, and `images/` folder in this directory.
2. `.env` must contain:
   ```
   CVAT_HOST=https://app.cvat.ai
   CVAT_USERNAME=your_username
   CVAT_PASSWORD=your_password
   #If applicable
   CVAT_ORG=your_org_name
   ```
3. Activate the venv: `source .venv/bin/activate`

## Workflow

1. Create a task in CVAT and upload your images there.
2. Copy the task ID from the CVAT URL (e.g. `https://app.cvat.ai/tasks/12345` → ID is `12345`).
3. Run inference and upload pre-annotations:
   ```bash
   python preannotate.py <task_id>
   ```

Open the task at https://app.cvat.ai → your task → Job #1 to verify and touch up.

## Tuning

Override defaults via CLI flags or `.env`:
- `--conf` / `CONF_THRESHOLD` (default 0.35) — lower = more polygons, more touch-up needed
- `--max-vertices` / `MAX_POLY_VERTICES` (default 150) — lower if CVAT UI lags
- `--no-enforce-lr` / `ENFORCE_LR=0` — disable left/right centroid swap
- `--model` / `MODEL_PATH` (default `best.pt`)
- `--images` / `IMAGE_DIR` (default `images/`)

Example:
```bash
python preannotate.py 12345 --conf 0.25 --max-vertices 100
```

## Re-running

`preannotate.py` OVERWRITES existing annotations on the task. Do not re-run after annotators start editing — create a new task instead.

For a new batch:
1. Replace contents of `images/`
2. Create a new task in CVAT, upload the new images, and run with the new task ID.
