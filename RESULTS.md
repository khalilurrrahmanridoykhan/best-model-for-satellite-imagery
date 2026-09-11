# Results

**These are smoke-test numbers, not real results.** Every notebook in this repo was
executed for real (not hand-written), but against a 20-chip sample (14 train / 3 val /
3 test) with a small epoch count, purely to prove the label pipeline and both training
loops run correctly end to end on real data. Read this as "the pipeline works," not
"here's how U-Net compares to YOLO on building detection."

## What was actually run

- Dataset: 20 chips from SpaceNet Buildings v2 (AOI_2_Vegas), downloaded via
  [`scripts/download_data.py`](scripts/download_data.py).
- Split: 14 train / 3 val / 3 test, by whole chip (`data/split.json`), shared by both
  models — no chip appears in more than one split.
- U-Net: `segmentation-models-pytorch` `Unet(encoder_name='resnet34',
  encoder_weights='imagenet')`, 5 epochs, batch size 4, Adam lr=1e-3,
  Dice+BCE combined loss (`notebooks/03_train_unet.ipynb`).
- YOLO: `ultralytics` `YOLO11n`, pretrained checkpoint, 15 epochs, imgsz=256
  (`notebooks/04_train_yolo.ipynb`).

## Numbers

| Model | Metric | Value | Epochs |
| :--- | :--- | :--- | :--- |
| U-Net | test IoU | 0.280 | 5 |
| U-Net | test F1 | 0.400 | 5 |
| YOLO | test mAP@0.5 | 0.075 | 15 |
| YOLO | test mAP@0.5:0.95 | 0.024 | 15 |

## A real finding worth keeping

At Ultralytics' default confidence threshold (0.25), the 15-epoch YOLO checkpoint
predicts **zero boxes** on all three held-out test chips — even though its own
mAP@0.5 is a nonzero 0.075. That's not a contradiction: mAP is computed by sweeping
the confidence threshold down to near zero, so it can credit a handful of correct
low-confidence boxes that a real deployment (using the default threshold) would never
actually surface. `notebooks/05_compare.ipynb` shows this explicitly rather than
quietly lowering the threshold to make the figures look better.

The practical takeaway: a benchmark metric being nonzero doesn't mean the model is
usable yet. Both checkpoints here need real training before either number means
anything.

## What "real" would need

- The full downloaded dataset (`python scripts/download_data.py --n 300` or more),
  not the 20-chip sample.
- Real epoch counts — enough that the training loss curves in `notebooks/03` and
  `notebooks/04` (`../runs/runs/yolo/results.csv`) actually flatten. `EPOCHS = 30+`
  for the U-Net, `EPOCHS = 50+` for YOLO are reasonable starting points, run on a
  free GPU notebook (Colab/Kaggle) rather than a local CPU.
- Re-running `notebooks/05_compare.ipynb` against those real checkpoints, and
  replacing the numbers and figures in this file with the real ones.

## Bugs caught by actually running this (not by inspection)

Documented in full in the relevant commit messages; summarized here since they
shaped why the pipeline looks the way it does:

1. **Non-polygon geometries in real SpaceNet labels.** 1 of 594 real labels sampled
   was a degenerate `Point`, and 3/594 were `MultiPolygon`. Neither is handled by
   naively assuming every label geometry is a `Polygon`. Fixed with
   `filter_polygon_geometries()` in `src/labels.py`, applied identically to both the
   U-Net and YOLO pipelines.
2. **Geographic vs. pixel coordinate mismatch.** `polygon_to_yolo_box` originally
   used raw polygon bounds without converting through the raster's affine transform
   — invisible against synthetic identity-transform test data, but produced
   nonsense boxes against real georeferenced chips. Fixed by requiring an explicit
   `transform` argument and inverting it.
3. **YOLO's `data.yaml` pointing train/val/test at the same directory.** Would have
   silently evaluated YOLO on its own training images. Fixed with per-split
   `data/yolo/{train,val,test}.txt` files.
4. **Those split files using paths relative to `data/yolo/`.** Ultralytics resolves
   each line against the working directory *at training time* (the notebook's own
   directory), not the `.txt` file's location — every image was reported
   corrupt/missing on the first real training attempt. Fixed by writing absolute
   paths.

## License

Apache-2.0
