"""
Runs the MSCKF on every extracted EuRoC Machine Hall sequence, evaluates
ATE against ground truth, and writes a JSON summary.
"""
import json
import os
import sys
import time

from run_eval import run as run_msckf
from evaluate import evaluate


DATA_ROOT = "/home/aryans-cat/Desktop/Studies/CV/Visual Inertial Odometry/Data/machine_hall"
RESULTS = "/home/aryans-cat/Desktop/Studies/CV/Visual Inertial Odometry/Results/Phase1"

# Per-dataset start offsets land the filter warm-up inside a static window
# (|a| ≈ 9.81, std ≲ 0.3 m/s²). The static period differs by sequence:
# MH_01 has ~40s on the ground; MH_02/03 have ~10–20s; MH_04/05 are static
# only for the first second or two before takeoff.
DATASET_OFFSETS = {
    "MH_01_easy":       40.0,
    "MH_02_easy":       10.0,
    "MH_03_medium":     10.0,
    "MH_04_difficult":   0.0,
    "MH_05_difficult":   0.0,
}
DATASETS = list(DATASET_OFFSETS.keys())


def dataset_path(name):
    return os.path.join(DATA_ROOT, name, name)


def main(ratio=1.0):
    os.makedirs(RESULTS, exist_ok=True)
    summary = {}
    t0 = time.time()
    for name in DATASETS:
        ds_path = dataset_path(name)
        if not os.path.isdir(os.path.join(ds_path, 'mav0')):
            print(f'SKIP {name} (not extracted)', file=sys.stderr)
            continue
        est_csv = os.path.join(RESULTS, f'{name}_est.csv')
        plot_png = os.path.join(RESULTS, f'{name}_traj.png')
        gt_csv = os.path.join(ds_path, 'mav0', 'state_groundtruth_estimate0', 'data.csv')

        print(f'\n=== {name} ===', file=sys.stderr)
        t_ds = time.time()
        offset = DATASET_OFFSETS[name]
        n, wall = run_msckf(ds_path, est_csv, ratio=ratio, offset=offset,
                            quiet=True)
        print(f'  {n} poses in {wall:.1f}s wall', file=sys.stderr)
        stats = evaluate(est_csv, gt_csv, plot_png=plot_png, align_scale=False, title=name)
        stats['poses'] = n
        stats['wall_s'] = wall
        summary[name] = stats
        print(f'  ATE RMSE {stats["rmse"]:.3f} m  mean {stats["mean"]:.3f} m  '
              f'max {stats["max"]:.3f} m  (n={stats["n"]})', file=sys.stderr)
        print(f'  elapsed {time.time()-t_ds:.1f}s', file=sys.stderr)

    with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nTOTAL wall: {time.time()-t0:.1f}s', file=sys.stderr)
    return summary


if __name__ == '__main__':
    main()
