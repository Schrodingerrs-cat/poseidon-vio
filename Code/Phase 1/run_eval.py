"""
Headless runner that executes the MSCKF pipeline on an EuRoC sequence and
dumps the estimated body-in-world trajectory to CSV.

Output CSV columns: timestamp,tx,ty,tz,qx,qy,qz,qw
"""
import argparse
import os
import sys
import time
import contextlib
from queue import Queue

from config import ConfigEuRoC
from dataset import EuRoCDataset, DataPublisher
from image import ImageProcessor
from msckf import MSCKF
from utils import to_quaternion


def run(path, out_csv, ratio=1.0, offset=40.0, duration=float('inf'),
        quiet=True):
    dataset = EuRoCDataset(path)
    dataset.set_starttime(offset=offset)

    img_queue = Queue()
    imu_queue = Queue()
    feature_queue = Queue()

    config = ConfigEuRoC()
    image_processor = ImageProcessor(config)
    msckf = MSCKF(config)

    poses = []

    def process_imu():
        while True:
            m = imu_queue.get()
            if m is None:
                return
            image_processor.imu_callback(m)
            msckf.imu_callback(m)

    def process_img():
        while True:
            m = img_queue.get()
            if m is None:
                feature_queue.put(None)
                return
            fm = image_processor.stareo_callback(m)
            if fm is not None:
                feature_queue.put(fm)

    def process_feature():
        while True:
            fm = feature_queue.get()
            if fm is None:
                return
            result = msckf.feature_callback(fm)
            if result is not None:
                R = result.pose.R
                t = result.pose.t
                q = to_quaternion(R)  # [qx,qy,qz,qw] JPL
                poses.append((result.timestamp, t[0], t[1], t[2],
                              q[0], q[1], q[2], q[3]))

    from threading import Thread
    t_imu = Thread(target=process_imu)
    t_img = Thread(target=process_img)
    t_feat = Thread(target=process_feature)
    t_imu.start(); t_img.start(); t_feat.start()

    imu_pub = DataPublisher(dataset.imu, imu_queue, duration, ratio)
    img_pub = DataPublisher(dataset.stereo, img_queue, duration, ratio)

    # Optionally silence MSCKF / ImageProcessor prints
    ctx = contextlib.redirect_stdout(open(os.devnull, 'w')) if quiet \
        else contextlib.nullcontext()

    with ctx:
        now = time.time()
        imu_pub.start(now)
        img_pub.start(now)
        t_imu.join()
        t_img.join()
        t_feat.join()

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w') as f:
        f.write('timestamp,tx,ty,tz,qx,qy,qz,qw\n')
        for row in poses:
            f.write(','.join(f'{v:.9f}' for v in row) + '\n')

    return len(poses), time.time() - now


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ratio', type=float, default=1.0)
    ap.add_argument('--offset', type=float, default=40.0)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    n, wall = run(args.path, args.out, ratio=args.ratio, offset=args.offset,
                  quiet=not args.verbose)
    print(f'[{os.path.basename(args.path)}] {n} poses in {wall:.1f}s -> {args.out}',
          file=sys.stderr)
