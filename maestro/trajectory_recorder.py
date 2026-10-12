"""DROID-format trajectory writer for data collection during trials.

Records a DROID-format trajectory using DROID's own utilities, so Maestro is fully standalone.
It records robot observations + actions at control rate into an HDF5 trajectory, and
(when ``post_process`` is on) also writes per-camera videos + frame JPEGs and a
``trajectory.npz`` of states/actions — the standard DROID data-collection outputs.
"""
import os
from collections import defaultdict
from copy import deepcopy
from queue import Empty, Queue

import h5py
import imageio
import numpy as np
from PIL import Image

from droid.misc.subprocess_utils import run_threaded_command
from droid.data_processing.timestep_processing import TimestepProcesser
from droid.calibration.calibration_utils import calib_info_filepath
from droid.camera_utils.info import camera_type_to_string_dict


def save_calibration_info(save_path):
    """Snapshot the current camera calibration next to a trajectory."""
    import shutil
    shutil.copyfile(calib_info_filepath, save_path)


def write_dict_to_hdf5(hdf5_file, data_dict, keys_to_ignore=["image", "depth", "pointcloud"]):
    for key in data_dict.keys():
        if key in keys_to_ignore:
            continue
        curr_data = data_dict[key]
        if type(curr_data) == list:
            curr_data = np.array(curr_data)
        dtype = type(curr_data)
        if dtype == dict:
            if key not in hdf5_file:
                hdf5_file.create_group(key)
            write_dict_to_hdf5(hdf5_file[key], curr_data)
            continue
        if key not in hdf5_file:
            if dtype != np.ndarray:
                dshape = ()
            else:
                dtype, dshape = curr_data.dtype, curr_data.shape
            hdf5_file.create_dataset(key, (1, *dshape), maxshape=(None, *dshape), dtype=dtype)
        else:
            hdf5_file[key].resize(hdf5_file[key].shape[0] + 1, axis=0)
        hdf5_file[key][-1] = curr_data


class TrajectoryWriter:
    """Threaded HDF5 trajectory writer (DROID format)."""

    def __init__(self, filepath, metadata=None, exists_ok=False, post_process=False):
        assert (not os.path.isfile(filepath)) or exists_ok
        self._dirpath = os.path.dirname(filepath)
        self._hdf5_file = h5py.File(filepath, "w")
        self._queue_dict = defaultdict(Queue)
        self._open = True

        self.post_process = post_process
        if self.post_process:
            image_transform_kwargs = {"remove_alpha": True, "bgr_to_rgb": True}
            self._timestep_processor = TimestepProcesser(
                ignore_action=True,  # we build the DROID-style action dict ourselves in write_timestep
                camera_extrinsics=["fixed_camera", "hand_camera", "varied_camera"],
                image_transform_kwargs=image_transform_kwargs,
            )
            os.makedirs(os.path.join(self._dirpath, "recordings"), exist_ok=True)
            self._video_writers = {}
            self._npz_data = {"states": [], "actions_pos": [], "actions_vel": []}
            self.t = {}

        if metadata is not None:
            self._update_metadata(metadata)

        def hdf5_writer(data):
            return write_dict_to_hdf5(self._hdf5_file, data)

        run_threaded_command(self._write_from_queue, args=(hdf5_writer, self._queue_dict["hdf5"]))

    def write_timestep(self, timestep):
        self._queue_dict["hdf5"].put(timestep)
        if self.post_process and not timestep["observation"]["timestamp"]["skip_action"]:
            raw_action = timestep["action"]
            processed = self._timestep_processor.forward(timestep)
            # forward() (ignore_action=True) omits the action; build the DROID-style
            # per-action-space dict (arm action + gripper) the .npz export expects.
            processed["action"] = {}
            for space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]:
                if space not in raw_action:
                    continue
                grip_key = "gripper_velocity" if "velocity" in space else "gripper_position"
                grip = raw_action.get(grip_key, raw_action.get("gripper_position"))
                processed["action"][space] = np.concatenate([raw_action[space], [grip]], dtype=np.float32)
            self._update_npz_data(processed)
            self._update_video_files(timestep)

    def _update_metadata(self, metadata):
        for key in metadata:
            self._hdf5_file.attrs[key] = deepcopy(metadata[key])

    def _write_from_queue(self, writer, queue):
        while self._open:
            try:
                data = queue.get(timeout=1)
            except Empty:
                continue
            writer(data)
            queue.task_done()

    def _update_video_files(self, timestep):
        # Keep physical camera identities: multiple cameras can share one type.
        observation = timestep["observation"]
        image_dict = observation.get("image", {})
        for serial, kind in sorted(observation["camera_type"].items()):
            img = image_dict.get(f"{serial}_left")
            if img is None:
                continue
            img = self._timestep_processor.image_transformer.transform(img)
            video_id = f"{camera_type_to_string_dict[kind]}_{serial}"
            if video_id not in self._video_writers:
                filename = os.path.join(self._dirpath, "recordings", f"{video_id}.mp4")
                self._video_writers[video_id] = imageio.get_writer(filename, fps=15, macro_block_size=1)
                run_threaded_command(
                    self._write_from_queue,
                    args=(self._video_writers[video_id].append_data, self._queue_dict[video_id]),
                )
            if video_id not in self.t:
                self.t[video_id] = 0
                os.makedirs(os.path.join(self._dirpath, "recordings", "frames", video_id), exist_ok=True)
            self._queue_dict[video_id].put(img)
            Image.fromarray(img[:, :, :3]).save(
                os.path.join(self._dirpath, "recordings", "frames", video_id, f"{self.t[video_id]:05d}.jpg")
            )
            self.t[video_id] += 1

    def _update_npz_data(self, timestep):
        self._npz_data["states"].append(timestep["observation"]["state"])
        self._npz_data["actions_pos"].append(timestep["action"]["cartesian_position"])
        self._npz_data["actions_vel"].append(timestep["action"]["cartesian_velocity"])

    def close(self, metadata=None):
        if metadata is not None:
            self._update_metadata(metadata)
        [queue.join() for queue in self._queue_dict.values()]
        if self.post_process:
            for video_id in self._video_writers:
                self._video_writers[video_id].close()
            self._video_writers.clear()
            self._npz_data = {k: np.array(v) for k, v in self._npz_data.items()}
            np.savez(os.path.join(self._dirpath, "trajectory.npz"), **self._npz_data)
        self._hdf5_file.close()
        self._open = False
