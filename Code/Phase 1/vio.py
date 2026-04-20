
from queue import Queue
from threading import Thread

from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF



class VIO(object):
    def __init__(self, config, img_queue, imu_queue, viewer=None):
        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            # print('img_msg', img_msg.timestamp)

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.stareo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return
            # print('imu_msg', imu_msg.timestamp)

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return
            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)

            if result is not None and self.viewer is not None:
                self.viewer.update_pose(result.cam0_pose)
        


if __name__ == '__main__':
    import time
    import argparse

    from dataset import EuRoCDataset, DataPublisher

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='path/to/your/EuRoC_MAV_dataset/MH_01_easy',
        help='Path of EuRoC MAV dataset.')
    parser.add_argument('--view', action='store_true', help='Show trajectory.')
    parser.add_argument('--gt', type=str, default=None,
        help='Optional GT CSV (EuRoC state_groundtruth_estimate0/data.csv) '
             'to overlay in red. Requires --est for the alignment reference.')
    parser.add_argument('--est', type=str, default=None,
        help='Prior estimate CSV used to compute SE(3) alignment of GT '
             'into the filter frame for visualization.')
    parser.add_argument('--offset', type=float, default=40.0,
        help='Seconds of the sequence to skip before starting the filter '
             '(static-init window).')
    parser.add_argument('--ratio', type=float, default=1.0,
        help='Playback speed multiplier (1.0 = real-time, <1 slower).')
    args = parser.parse_args()

    gt_traj = None
    if args.gt and args.est:
        import numpy as _np
        from evaluate import load_est, load_gt, associate, umeyama
        _est_ts, _est_P = load_est(args.est)
        _gt_ts, _gt_P = load_gt(args.gt)
        _pairs = associate(_est_ts, _gt_ts, max_dt=0.02)
        _est_m = _np.array([_est_P[i] for i, _ in _pairs])
        _gt_m = _np.array([_gt_P[j] for _, j in _pairs])
        # Umeyama (without scale): s*R@est + t = gt  -->  est = R.T @ (gt - t)
        _s, _R, _t = umeyama(_est_m, _gt_m, with_scale=False)
        gt_traj = ((_gt_P - _t) @ _R) / _s

    if args.view:
        try:
            from viewer import Viewer
        except Exception:
            # pangolin not available — use matplotlib/cv2 fallback
            from viewer_mpl import Viewer
        viewer = Viewer(gt_trajectory=gt_traj)
    else:
        viewer = None

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=args.offset)


    img_queue = Queue()
    imu_queue = Queue()

    config = ConfigEuRoC()
    msckf_vio = VIO(config, img_queue, imu_queue, viewer=viewer)


    duration = float('inf')
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, args.ratio)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, args.ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)