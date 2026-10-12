from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import imageio
import numpy as np

from maestro.trajectory_recorder import TrajectoryWriter


class TrajectoryRecorderTests(unittest.TestCase):
    def test_hdf5_npz_video_and_frames(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = TrajectoryWriter(str(root / 'trajectory.h5'), post_process=True)
            observation = {
                'timestamp': {'skip_action': False},
                'robot_state': {'cartesian_position': np.zeros(6), 'gripper_position': 0.0,
                                'joint_positions': np.zeros(7), 'joint_velocities': np.zeros(7)},
                'camera_type': {'test_camera': 0},
                'camera_extrinsics': {'test_camera_left': np.zeros(6)},
                'camera_intrinsics': {'test_camera_left': np.eye(3)},
                'image': {'test_camera_left': np.full((48, 64, 4), 128, dtype=np.uint8)},
            }
            action = {'cartesian_position': np.arange(6, dtype=float),
                      'cartesian_velocity': np.zeros(6), 'gripper_position': 0.25,
                      'gripper_velocity': 0.0}
            try:
                for _ in range(3):
                    writer.write_timestep({'observation': observation, 'action': action})
            finally:
                writer.close()
            with h5py.File(root / 'trajectory.h5', 'r') as data:
                self.assertEqual(data['action/cartesian_position'].shape, (3, 6))
                np.testing.assert_array_equal(data['action/gripper_position'][:], [0.25] * 3)
            with np.load(root / 'trajectory.npz') as data:
                self.assertEqual(data['states'].shape[0], 3)
                self.assertEqual(data['actions_pos'].shape, (3, 7))
                np.testing.assert_array_equal(data['actions_pos'][:, -1], [0.25] * 3)
            self.assertEqual(len(list((root / 'recordings/frames/hand_camera_test_camera').glob('*.jpg'))), 3)
            with imageio.get_reader(root / 'recordings/hand_camera_test_camera.mp4') as video:
                self.assertEqual(video.count_frames(), 3)
                self.assertEqual(video.get_data(0).shape, (48, 64, 3))

    def test_two_external_cameras_keep_distinct_videos(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = TrajectoryWriter(str(root / 'trajectory.h5'), post_process=True)
            cameras = {'11111111': 0, '22222222': 1, '33333333': 1}
            colors = {'11111111': [255, 0, 0], '22222222': [0, 255, 0],
                      '33333333': [0, 0, 255]}
            images = {}
            for serial, color in colors.items():
                images[f'{serial}_left'] = np.tile(color + [255], (48, 64, 1)).astype(np.uint8)
                images[f'{serial}_right'] = np.zeros((48, 64, 4), dtype=np.uint8)
            observation = {
                'timestamp': {'skip_action': False},
                'robot_state': {'cartesian_position': np.zeros(6), 'gripper_position': 0.0,
                                'joint_positions': np.zeros(7), 'joint_velocities': np.zeros(7)},
                'camera_type': cameras,
                'camera_extrinsics': {key: np.zeros(6) for key in images},
                'camera_intrinsics': {key: np.eye(3) for key in images},
                'image': images,
            }
            action = {'cartesian_position': np.zeros(6), 'cartesian_velocity': np.zeros(6),
                      'gripper_position': 0.0, 'gripper_velocity': 0.0}
            try:
                for _ in range(3):
                    writer.write_timestep({'observation': observation, 'action': action})
            finally:
                writer.close()
            self.assertEqual(len(list((root / 'recordings').glob('*.mp4'))), 3)
            for serial, kind in cameras.items():
                name = f'{"hand_camera" if kind == 0 else "varied_camera"}_{serial}'
                frames = sorted((root / 'recordings/frames' / name).glob('*.jpg'))
                self.assertEqual(len(frames), 3)
                with imageio.get_reader(root / 'recordings' / f'{name}.mp4') as video:
                    self.assertEqual(video.count_frames(), 3)
                    frame = video.get_data(0)
                    np.testing.assert_allclose(frame.mean(axis=(0, 1)), colors[serial][::-1], atol=5)
                np.testing.assert_array_equal(images[f'{serial}_left'][0, 0], colors[serial] + [255])


if __name__ == '__main__':
    unittest.main()
