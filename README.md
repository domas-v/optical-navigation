# README

Dependencies:
- python 3.12
- opencv
- numpy
- matplotlib

## Install

Can install with `uv sync` and `uv sync --group dev`.
Or just `pip install -r requirements.txt`

## Run

```bash
python odometry.py --video path/to/video.mp4 --log path/to/logs.csv
```

Add `--no-display` to skip the OpenCV preview window.

## Output

- **`output.mp4`** (annotated video) 
- **`odometry.npz`** (per-frame flow and calibration metadata). Can then run `python plot_dist.py --log path/to/logs.csv --npz odometry.npz`
to plot error.


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