# U-Net vs YOLO — Same-Dataset Practice

A focused practice project: train **both** a U-Net (semantic segmentation) and a
**YOLO** (object detection) model on the **same source labels**, so the comparison
between them is apples-to-apples — same imagery, same real-world objects (building
footprints), two different ways of "seeing" them.

- **U-Net** answers: *which pixels are building?*
- **YOLO** answers: *where are the individual buildings, as countable objects?*

Same building-footprint polygons feed both: rasterized into pixel masks for U-Net,
converted into bounding boxes for YOLO. That shared origin — and where the two models'
outputs then diverge — is the actual point of this repo, not just "trained two models."

## Dataset

[SpaceNet Buildings v2](https://spacenet.ai/datasets/) (AOI_2_Vegas) — free, public,
one building-footprint polygon per building already provided. Individual chips and
labels are pulled directly from the public `spacenet-dataset` S3 bucket (anonymous
access, no AWS account needed) by [`scripts/download_data.py`](scripts/download_data.py),
rather than downloading the official 25.6GB training tarball. Raw imagery is not
committed to this repo (see `.gitignore`) since it's fully reproducible from that
script.

## Repo layout

```
data/
├── raw/          # downloaded SpaceNet chips + GeoJSON labels (gitignored)
├── unet/         # rasterized masks, built from data/raw/ by notebooks/02
└── yolo/         # YOLO-format boxes, built from data/raw/ by notebooks/02
notebooks/        # the actual pipeline, in order (01 -> 05)
src/              # shared, unit-tested code the notebooks import
tests/            # pytest tests for src/, using synthetic data (no download needed)
scripts/          # data download
```

## Pipeline

| Notebook | Does |
| :--- | :--- |
| `01_explore_data.ipynb` | Load a few chips + polygons, sanity-check alignment |
| `02_build_labels.ipynb` | Polygons -> U-Net masks *and* YOLO boxes, from one source; the shared spatial train/val/test split |
| `03_train_unet.ipynb` | Train U-Net (`segmentation-models-pytorch`) -- run on a free GPU (Colab/Kaggle) |
| `04_train_yolo.ipynb` | Train YOLO (`ultralytics`) -- run on a free GPU (Colab/Kaggle) |
| `05_compare.ipynb` | Same held-out chips through both models; the honest comparison |

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Unit tests for the label pipeline -- no dataset download needed, uses synthetic data
pytest tests/ -v

# The real dataset (needed for the notebooks) -- downloads N chips + their
# GeoJSON labels from the public SpaceNet S3 bucket
python scripts/download_data.py --n 300
```

Then run the notebooks in order (`jupyter notebook` or `jupyter nbconvert --execute`).
`03` and `04` (the actual training) are meant to run on a free GPU notebook (Google
Colab or Kaggle) with the full downloaded dataset and a real epoch count — the
versions committed to this repo were smoke-tested locally on a 20-chip sample with a
handful of epochs each, purely to prove the pipeline runs correctly end to end. See
[`RESULTS.md`](RESULTS.md) for exactly what that means and what a real run still needs.

## Results

See [`RESULTS.md`](RESULTS.md) — smoke-test numbers only, not real results yet.

## Why this exists

Part of a larger, ongoing geospatial-AI-for-public-health learning track. This repo is
the standalone warm-up: same dataset, same labels, two model families, one clean
comparison -- before applying both to a real dengue/flood risk-mapping pipeline.

## License

Apache-2.0
