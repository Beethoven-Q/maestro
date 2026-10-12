"""Check a ZED camera, and optionally the depth service, without connecting to the robot.

    uv run python scripts/check_camera_depth.py --output /tmp/camera_check --depth

Saves the left and right images and a report (camera matrix, baseline, depth percentiles).
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyzed.sl as sl

from maestro.camera_geometry import stereo_geometry
from maestro.config import cfg
from maestro.perception_utils import stereo_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', default=cfg.cameras.hand_camera_id, help='ZED serial (default: the wrist camera)')
    parser.add_argument('--output', type=Path, required=True, help='folder for the images and report')
    parser.add_argument('--depth', action='store_true', help='also send the pair to the FoundationStereo service')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    camera = sl.Camera()
    params = sl.InitParameters()
    params.set_from_serial_number(int(args.serial))
    params.camera_resolution = sl.RESOLUTION.HD720
    params.camera_fps = 15
    params.coordinate_units = sl.UNIT.METER
    params.depth_mode = sl.DEPTH_MODE.NONE
    status = camera.open(params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f'Could not open camera {args.serial}: {status}')
    try:
        for _ in range(15):
            status = camera.grab()
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f'Camera capture failed: {status}')
        images = {}
        for name, view in [('left', sl.VIEW.LEFT), ('right', sl.VIEW.RIGHT)]:
            frame = sl.Mat()
            status = camera.retrieve_image(frame, view)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f'Image retrieval failed: {status}')
            images[name] = cv2.cvtColor(frame.get_data(), cv2.COLOR_BGRA2RGB)
            if not cv2.imwrite(str(args.output / f'{name}.png'), cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f'Could not save {name} image')
        calibration = camera.get_camera_information().camera_configuration.calibration_parameters
        matrix, baseline = stereo_geometry(calibration)
        report = {'serial': str(args.serial), 'sdk': camera.get_sdk_version(),
                  'shape': list(images['left'].shape[:2]), 'K': matrix.tolist(),
                  'baseline_m': baseline, 'robot_connected': False}
    finally:
        camera.close()

    if args.depth:
        start = time.monotonic()
        depth = stereo_depth(images['left'], images['right'], calibration)
        np.save(args.output / 'depth.npy', depth)
        report['depth'] = {'seconds': time.monotonic() - start,
                           'percentiles_m': np.percentile(depth, [5, 50, 95]).tolist()}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
