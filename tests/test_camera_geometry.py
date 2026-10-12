from types import SimpleNamespace
import unittest

import numpy as np

from maestro.camera_geometry import stereo_geometry


class StereoGeometryTests(unittest.TestCase):
    def calibration(self, baseline=0.063):
        return SimpleNamespace(
            left_cam=SimpleNamespace(fx=700, fy=701, cx=640, cy=360),
            get_camera_baseline=lambda: baseline,
        )

    def test_metric_camera_geometry_is_read(self):
        matrix, baseline = stereo_geometry(self.calibration())
        np.testing.assert_array_equal(matrix, [[700, 0, 640], [0, 701, 360], [0, 0, 1]])
        self.assertAlmostEqual(baseline, 0.063)

    def test_invalid_or_millimeter_baseline_is_rejected(self):
        for baseline in [0, 63, float('nan'), -0.12]:
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                stereo_geometry(self.calibration(baseline))

    def test_invalid_focal_length_is_rejected(self):
        calibration = self.calibration()
        calibration.left_cam.fx = float('nan')
        with self.assertRaises(ValueError):
            stereo_geometry(calibration)


if __name__ == '__main__':
    unittest.main()
