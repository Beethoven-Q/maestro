import os
import sys
import threading
from copy import deepcopy

import cv2
import numpy as np

from droid.camera_utils.info import get_camera_name
from droid.misc.parameters import hand_camera_id
from droid.misc.time import time_ms

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: You have not setup the ZED cameras, and currently cannot use them")


def gather_zed_cameras():
    all_zed_cameras = []
    try:
        cameras = sl.Camera.get_device_list()
    except NameError as exc:
        raise RuntimeError("ZED Python API is unavailable; install the matching pyzed wheel.") from exc

    for cam in cameras:
        cam = ZedCamera(cam)
        all_zed_cameras.append(cam)

    return all_zed_cameras


resize_func_map = {"cv2": cv2.resize, None: None}

# Grabs to try before reopening the camera. A dropped USB link fails each grab only
# after the SDK's timeout (about 3 s).
GRAB_ATTEMPTS = 3


class ZedCamera:
    def __init__(self, camera):
        # Save Parameters #
        self.serial_number = str(camera.serial_number)
        self.is_hand_camera = self.serial_number == hand_camera_id
        self.high_res_calibration = False
        self.current_mode = None
        self._current_params = None
        self._extriniscs = {}
        # Tool calls, the video recorder thread and the DataRecorder all read this camera.
        self._lock = threading.RLock()
        self._failure = None
        self._recording_file = None
        self._recording_segments = 0
        flip = sl.FLIP_MODE.OFF if self.is_hand_camera else sl.FLIP_MODE.ON
        self.standard_params = dict(
            depth_minimum_distance=0.1, camera_resolution=sl.RESOLUTION.HD720, depth_stabilization=False, camera_fps=60, camera_image_flip=flip
        )

        self.advanced_params = dict(
            depth_minimum_distance=0.1, camera_resolution=sl.RESOLUTION.HD2K, depth_stabilization=False, camera_fps=15, camera_image_flip=flip
        )
        # Open Camera #
        print("Opening Zed: ", self.serial_number)

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image=True,
        depth=True,
        # depth=False,
        pointcloud=True,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        # Non-Permenant Values #
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution

        # Permenant Values #
        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    ### Camera Modes ###
    def set_calibration_mode(self):
        with self._lock:
            # Set Parameters #
            self.image = True
            self.concatenate_images = False
            self.skip_reading = False
            self.zed_resolution = sl.Resolution(0, 0)
            self.resizer_resolution = (0, 0)

            # Set Mode #
            change_settings_1 = (self.high_res_calibration) and (self._current_params != self.advanced_params)
            change_settings_2 = (not self.high_res_calibration) and (self._current_params != self.standard_params)
            if change_settings_1:
                self._configure_camera(self.advanced_params)
            if change_settings_2:
                self._configure_camera(self.standard_params)
            self.current_mode = "calibration"

    def set_trajectory_mode(self):
        with self._lock:
            # Set Parameters #
            self.image = self.traj_image
            self.concatenate_images = self.traj_concatenate_images
            self.skip_reading = not any([self.image, self.depth, self.pointcloud])

            if self.resize_func is None:
                self.zed_resolution = sl.Resolution(*self.traj_resolution)
                self.resizer_resolution = (0, 0)
            else:
                self.zed_resolution = sl.Resolution(0, 0)
                self.resizer_resolution = self.traj_resolution

            # Set Mode #
            change_settings = self._current_params != self.standard_params
            if change_settings:
                self._configure_camera(self.standard_params)
            self.current_mode = "trajectory"

    def _configure_camera(self, init_params):
        # Close Existing Camera #
        self.disable_camera()

        # Initialize Readers #
        self._cam = sl.Camera()
        self._sbs_img = sl.Mat()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._left_depth = sl.Mat()
        self._right_depth = sl.Mat()
        self._left_pointcloud = sl.Mat()
        self._right_pointcloud = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        # Open Camera #
        self._current_params = init_params
        sl_params = sl.InitParameters(**init_params)
        sl_params.set_from_serial_number(int(self.serial_number))
        sl_params.camera_image_flip = sl.FLIP_MODE.OFF if self.is_hand_camera else sl.FLIP_MODE.ON
        sl_params.depth_mode = sl.DEPTH_MODE.ULTRA
        sl_params.coordinate_units = sl.UNIT.METER
        status = self._cam.open(sl_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED {self.serial_number} failed to open: {status}. Check USB3, permissions, and other camera processes.")
        self._failure = None

        # Save Intrinsics #
        self.latency = int(2.5 * (1e3 / sl_params.camera_fps))
        calib_params = self._cam.get_camera_information().camera_configuration.calibration_parameters
        self._intrinsics = {
            self.serial_number + "_left": self._process_intrinsics(calib_params.left_cam),
            self.serial_number + "_right": self._process_intrinsics(calib_params.right_cam),
        }

    def _reopen(self):
        """Close and reopen the camera in its current mode, resuming SVO recording in a new file."""
        mode, recording_file = self.current_mode, self._recording_file
        try:
            self._configure_camera(self._current_params)
        finally:
            self.current_mode = mode
        if recording_file is None:
            return
        self._recording_segments += 1
        base, ext = os.path.splitext(recording_file)
        segment_file = f"{base}_{self._recording_segments}{ext}"
        err = self._cam.enable_recording(sl.RecordingParameters(segment_file, sl.SVO_COMPRESSION_MODE.H265))
        if err == sl.ERROR_CODE.SUCCESS:
            self._recording_file = recording_file
        else:
            self._warn(f"SVO recording not resumed ({err})")

    def _warn(self, message):
        # stderr reaches the operator live; a tool's stdout is captured for the planner.
        print(f"ZED {self.serial_number} ({get_camera_name(self.serial_number)}): {message}", file=sys.stderr)

    ### Calibration Utilities ###
    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])
        intrinsics["distCoeffs"] = np.array(list(params.disto))
        return intrinsics

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    ### Recording Utilities ###
    def start_recording(self, filename):
        assert filename.endswith(".svo")
        with self._lock:
            recording_param = sl.RecordingParameters(filename, sl.SVO_COMPRESSION_MODE.H265)
            err = self._cam.enable_recording(recording_param)
            assert err == sl.ERROR_CODE.SUCCESS
            self._recording_file, self._recording_segments = filename, 0

    def stop_recording(self):
        with self._lock:
            self._recording_file = None
            self._cam.disable_recording()

    ### Basic Camera Utilities ###
    def _process_frame(self, frame):
        frame = deepcopy(frame.get_data())
        if self.resizer_resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resizer_resolution)

    def _grab(self):
        """Grab a frame, retrying and then reopening the camera once before failing.

        grab() does not reattach a camera that dropped off the USB bus; reopening it by
        serial can. A failure that survives the reopen is latched: pyzed holds the GIL
        during grab and open, so readers retrying a dead camera would stall the process.
        """
        if self._failure:
            raise RuntimeError(self._failure)
        for attempt in range(1, GRAB_ATTEMPTS + 1):
            err = self._cam.grab(self._runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                return
            self._warn(f"grab failed ({err}), attempt {attempt}/{GRAB_ATTEMPTS}")
        self._warn("reopening camera")
        try:
            self._reopen()
            err = self._cam.grab(self._runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self._warn("recovered after reopening")
                return
            self._warn(f"grab failed after reopening ({err})")
        except RuntimeError as exc:
            self._warn(str(exc))
        self._failure = (
            f"ZED {self.serial_number} ({get_camera_name(self.serial_number)}) stopped delivering frames "
            "and did not recover after a reopen. Check its USB connection, then restart Maestro."
        )
        raise RuntimeError(self._failure)

    def get_calibration(self):
        """Stereo calibration of the open camera; raises the camera's failure if it stopped."""
        with self._lock:
            if self._failure:
                raise RuntimeError(self._failure)
            return self._cam.get_camera_information().camera_configuration.calibration_parameters

    def read_camera(self):
        with self._lock:
            # Skip if Read Unnecesary #
            if self.skip_reading or not self.is_running():
                return {}, {}

            # Read Camera #
            timestamp_dict = {self.serial_number + "_read_start": time_ms()}
            self._grab()
            timestamp_dict[self.serial_number + "_read_end"] = time_ms()

            # Benchmark Latency #
            received_time = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            timestamp_dict[self.serial_number + "_frame_received"] = received_time
            timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

            # Return Data #
            data_dict = {}

            if self.image:
                if self.concatenate_images:
                    self._cam.retrieve_image(self._sbs_img, sl.VIEW.SIDE_BY_SIDE, resolution=self.zed_resolution)
                    data_dict["image"] = {self.serial_number: self._process_frame(self._sbs_img)}
                else:
                    self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT, resolution=self.zed_resolution)
                    self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT, resolution=self.zed_resolution)
                    data_dict["image"] = {
                        self.serial_number + "_left": self._process_frame(self._left_img),
                        self.serial_number + "_right": self._process_frame(self._right_img),
                    }
            if self.depth:
                self._cam.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH, resolution=sl.Resolution(*self.resizer_resolution))
                data_dict['depth'] = {self.serial_number + '_left': self._left_depth.get_data().copy()}
            if self.pointcloud:
                self._cam.retrieve_measure(self._left_pointcloud, sl.MEASURE.XYZRGBA, resolution=sl.Resolution(*self.resizer_resolution))
                data_dict['pointcloud'] = {
                    self.serial_number + '_left': self._left_pointcloud.get_data().copy(),
                    self.serial_number + '_right': self._right_pointcloud.get_data().copy()
                }

            return data_dict, timestamp_dict

    def disable_camera(self):
        with self._lock:
            if self.current_mode == "disabled":
                return
            if hasattr(self, "_cam"):
                self._current_params = None
                self._recording_file = None  # closing ends any SVO recording
                self._cam.close()
            self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"
