import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import requests

from maestro import vlm
from maestro.api import SceneScanCache
from maestro.perception_utils import fuse_observation, segment, stereo_depth


class PerceptionContractTests(unittest.TestCase):
    @patch('maestro.vlm.PROVIDER', 'gemini')
    @patch('maestro.vlm._gemini')
    def test_pointing_jpeg_preserves_red_channel(self, gemini):
        generate = gemini.return_value.models.generate_content
        generate.return_value = SimpleNamespace(text='[]')
        red = np.zeros((32, 32, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        vlm.point([red], ['red cube'])
        encoded = generate.call_args.kwargs['contents'][0].inline_data.data
        decoded = np.asarray(Image.open(io.BytesIO(encoded)))
        self.assertGreater(decoded[:, :, 0].mean(), 240)
        self.assertLess(decoded[:, :, 2].mean(), 15)

    def test_scan_cache_serializes_numeric_intrinsics(self):
        with TemporaryDirectory() as tmp:
            cache = SceneScanCache(tmp)
            cache.set([{'left_image': np.zeros((8, 8, 3), dtype=np.uint8),
                        'right_image': np.zeros((8, 8, 3), dtype=np.uint8),
                        'depth': np.ones((8, 8)), 'target_joints': np.zeros(7),
                        'intrinsics': {'cameraMatrix': np.eye(3), 'distCoeffs': np.zeros(5)},
                        'cam_pose': np.eye(4)}])
            cache.save_to_disk()
            metadata = json.loads((Path(tmp) / 'meta.json').read_text())
            np.testing.assert_array_equal(metadata[0]['intrinsics'], np.eye(3))

    @patch('maestro.perception_utils.requests.post')
    def test_stereo_depth_sends_pair_and_geometry_and_validates_output(self, post):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        calibration = SimpleNamespace(left_cam=SimpleNamespace(fx=700, fy=701, cx=4, cy=4),
                                      get_camera_baseline=lambda: 0.063)
        output = np.full((8, 8), 0.5, dtype=np.float32)

        def respond(*args, **kwargs):
            buffer = io.BytesIO()
            np.save(buffer, output)
            return Mock(content=buffer.getvalue())

        post.side_effect = respond
        np.testing.assert_array_equal(stereo_depth(image, image, calibration), output)
        request = post.call_args.kwargs
        self.assertTrue(post.call_args.args[0].endswith('/depth'))
        self.assertEqual(request['data'], {'fx': 700.0, 'baseline': 0.063})
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(request['files']['left'][1]))), image)
        output[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, 'invalid depth'):
            stereo_depth(image, image, calibration)
        post.side_effect = requests.HTTPError('depth service unavailable')
        with self.assertRaises(requests.HTTPError):
            stereo_depth(image, image, calibration)

    @patch('maestro.perception_utils.requests.post')
    def test_segmentation_without_detections_is_empty(self, post):
        buffer = io.BytesIO()
        np.savez(buffer, masks=np.zeros((0, 8, 8), bool), boxes=np.zeros((0, 4), np.float32),
                 scores=np.zeros(0, np.float32), labels=np.array([], dtype=str))
        post.return_value.content = buffer.getvalue()
        detections = segment(np.zeros((8, 8, 3), dtype=np.uint8), 'red cube')
        self.assertEqual(post.call_args.kwargs['data'], {'prompt': 'red cube'})
        self.assertEqual(detections['masks'].shape, (0, 8, 8))

    VIEW = {'depth': np.ones((4, 4)), 'intrinsics': np.eye(3), 'cam_pose': np.eye(4)}

    @patch('maestro.perception_utils.requests.post')
    def test_fusion_sends_views_in_the_request(self, post):
        post.return_value.json.return_value = {'success': True}
        fuse_observation([self.VIEW, self.VIEW])
        name, content, _ = post.call_args.kwargs['files']['views']
        with np.load(io.BytesIO(content)) as views:
            self.assertEqual(views['depths'].shape, (2, 4, 4))
            self.assertEqual(views['camera_poses'].shape, (2, 4, 4))

    @patch('maestro.perception_utils.requests.post')
    def test_fusion_application_error_stops_planning(self, post):
        post.return_value.json.return_value = {'success': False, 'error': 'invalid camera data'}
        with self.assertRaisesRegex(RuntimeError, 'Scene fusion failed'):
            fuse_observation([self.VIEW])

    @patch('maestro.perception_utils.requests.post')
    def test_fusion_http_error_stops_planning(self, post):
        post.return_value.raise_for_status.side_effect = requests.HTTPError('server error')
        with self.assertRaises(requests.HTTPError):
            fuse_observation([self.VIEW])

if __name__ == '__main__':
    unittest.main()
