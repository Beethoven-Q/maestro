"""Camera matrices and the rectified stereo geometry of the currently open camera."""
import numpy as np


def camera_matrix(intrinsics):
    """Normalize a DROID intrinsics record or a numeric 3x3 matrix."""
    if isinstance(intrinsics, dict):
        intrinsics = intrinsics['cameraMatrix']
    matrix = np.asarray(intrinsics, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Camera intrinsics must be a finite 3x3 matrix')
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError('Camera focal lengths must be positive')
    return matrix


def stereo_geometry(calibration):
    """Left camera matrix and baseline (meters) of an open ZED's rectified stereo pair."""
    camera = calibration.left_cam
    matrix = camera_matrix([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]])
    baseline = float(calibration.get_camera_baseline())
    if not np.isfinite(baseline) or not 0.01 < baseline < 1.0:
        raise ValueError(f"Invalid stereo baseline {baseline}; the ZED must use UNIT.METER")
    return matrix, baseline
