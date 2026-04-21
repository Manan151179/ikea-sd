# SUN RGB-D Raw Data

This directory holds the SUN RGB-D metadata files required by `src/data_parser.py`.
All contents are **intentionally excluded from git** (see `/.gitignore`).

---

## What you need

| File | Size | Required |
|------|------|----------|
| `SUNRGBDMeta2DBB_v2.mat` | ~4.3 MB | Yes — 2D bounding box metadata |
| `SUNRGBDMeta3DBB_v2.mat` | ~12 MB | Optional — 3D bounding box metadata |
| `SUNRGBD/` | ~6.4 GB | Yes — raw RGB-D scene tree (images + annotations) |

---

## Download instructions

### Step 1 — Metadata .mat files (small, fast)

```bash
# From the Princeton server (requires accepting their terms of use)
# Visit: https://rgbd.cs.princeton.edu and download:
#   SUNRGBD.zip  (full dataset, 6.4 GB)
#   SUNRGBDMeta2DBB_v2.mat
#   SUNRGBDMeta3DBB_v2.mat
```

Alternatively, the data_parser does NOT require the .mat files for the
file-system scan mode — they are only needed for `layout_dims` parsing.
You can run with `--skip-mat` and layout_dims will be null for all records.

### Step 2 — Full scene tree

```bash
# Option A: full download via wget (replace URL with current Princeton link)
wget -O /tmp/SUNRGBD.zip "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
unzip /tmp/SUNRGBD.zip -d data/raw/

# Option B: Google Drive mirror (if you have access)
# Place the unzipped SUNRGBD/ folder at:
#   /content/drive/MyDrive/ikea-sd/data/raw/SUNRGBD/
# notebook 01 will symlink it automatically.
```

### Step 3 — Verify structure

After download, `data/raw/` should look like:

```
data/raw/
├── SUNRGBDMeta2DBB_v2.mat
├── SUNRGBDMeta3DBB_v2.mat
└── SUNRGBD/
    ├── kv1/
    │   └── NYUdata/
    │       ├── NYU0001/
    │       │   ├── image/          ← RGB image
    │       │   ├── scene.txt       ← room type label
    │       │   ├── annotation2Dfinal/index.json   ← object labels
    │       │   └── annotation3Dlayout/layout.mat  ← floor-plan dims
    │       └── NYU0002/ ...
    ├── kv2/
    ├── realsense/
    └── xtion/
```

### Step 4 — Parse

```bash
python -m src.data_parser \
    --raw-root data/raw \
    --out data/processed/structured_scenes.json \
    --limit 50
```

---

## Notes

- The raw dataset is **~6.4 GB** — do not commit it.
- On Colab, notebook 01 symlinks the Drive copy to avoid duplicating the data.
- `data/processed/structured_scenes.json` (the parsed output) **is** tracked by git when small enough (≤50 scenes → ~30 KB).
