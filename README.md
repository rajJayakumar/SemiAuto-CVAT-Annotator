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
