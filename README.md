# README

## Install

Requires **Python 3.12+**. With [uv](https://github.com/astral-sh/uv), from the repo root run `uv sync` to install runtime dependencies (`numpy`, `opencv-python`).
For distance plotting (`plot_dist.py`), install dev tools too: `uv sync --group dev` (adds `matplotlib` and other dev-only packages).

## Run

```bash
uv run python odometry.py --video path/to/video.mp4 --log path/to/logs.csv
```

Add `--no-display` to skip the OpenCV preview window. Outputs are **`output.mp4`** (annotated video) and **`odometry.npz`** (per-frame flow and calibration metadata).

Can plot GPS vs odometry cumulative distance after running `odometry.py`

```bash
uv run --group dev python plot_dist.py --log path/to/logs.csv
```

## Logic

1. Load logs.
2. Interpolate logs.
3. Calculate camera intrinsics (K, K_inv)
4. For each frame:
    1. Process frame (rectify, mask)
    2. Track features with LK
    3. Compute distance
    4. Post process (draw stats and write to disk)
5. Save some data to .npz for plotting