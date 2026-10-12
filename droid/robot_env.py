from copy import deepcopy

import gym
import numpy as np

from droid.calibration.calibration_utils import load_calibration_info
from droid.camera_utils.info import camera_type_dict
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from droid.misc.parameters import hand_camera_id, nuc_ip
from droid.misc.server_interface import ServerInterface
from droid.misc.time import time_ms
from droid.misc.transformations import change_pose_frame


class RobotEnv(gym.Env):
    def __init__(self, action_space="cartesian_velocity", gripper_action_space=None, camera_kwargs={}, do_reset=True, verbose=False):
        # Initialize Gym Environment
        super().__init__()

        # Store verbose flag
        self.verbose = verbose
        
        if self.verbose:
            print("RobotEnv: Initializing robot environment...")

        # Define Action Space #
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.check_action_range = "velocity" in action_space

        if self.verbose:
            print(f"RobotEnv: Action space set to {action_space}, gripper action space: {gripper_action_space}")

        # Robot Configuration
        self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
        self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        self.DoF = 7 if ("cartesian" in action_space) else 8
        self.control_hz = 15

        if self.verbose:
            print(f"RobotEnv: Robot configuration - DoF: {self.DoF}, control_hz: {self.control_hz}")

        if nuc_ip is None:
            from droid.franka.robot import FrankaRobot

            if self.verbose:
                print("RobotEnv: Initializing FrankaRobot (local connection)")
            self._robot = FrankaRobot()
        else:
            if self.verbose:
                print(f"RobotEnv: Initializing ServerInterface with IP: {nuc_ip}")
            self._robot = ServerInterface(ip_address=nuc_ip)

        # Create Cameras
        if self.verbose:
            print("RobotEnv: Setting up cameras and calibration...")
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if do_reset:
            if self.verbose:
                print("RobotEnv: Performing initial robot reset...")
            self.reset()
        
        if self.verbose:
            print("RobotEnv: Initialization complete!")

    def step(self, action):
        if self.verbose:
            print(f"RobotEnv.step: Executing action with {len(action)} elements")
        
        # Check Action
        assert len(action) == self.DoF
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)
            if self.verbose:
                print(f"RobotEnv.step: Action range validated - min: {action.min():.3f}, max: {action.max():.3f}")

        # Update Robot
        if self.verbose:
            print(f"RobotEnv.step: Updating robot with action space: {self.action_space}")
        action_info = self.update_robot(
            action,
            action_space=self.action_space,
            gripper_action_space=self.gripper_action_space,
        )

        if self.verbose:
            print("RobotEnv.step: Action execution complete")

        # Return Action Info
        return action_info

    def reset(self, randomize=False):
        if self.verbose:
            print(f"RobotEnv.reset: Resetting robot (randomize={randomize})")
        
        if self.verbose:
            print("RobotEnv.reset: Updating gripper to open position")
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
            if self.verbose:
                print(f"RobotEnv.reset: Generated random noise: {noise}")
        else:
            noise = None
            if self.verbose:
                print("RobotEnv.reset: No randomization applied")

        if self.verbose:
            print("RobotEnv.reset: Moving to reset joint positions")
        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)
        
        if self.verbose:
            print("RobotEnv.reset: Reset complete")

    def update_robot(self, action, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        if self.verbose:
            print(f"RobotEnv.update_robot: Sending command with action_space={action_space}, blocking={blocking}")
        
        action_info = self._robot.update_command(
            action,
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            blocking=blocking
        )
        
        if self.verbose:
            print("RobotEnv.update_robot: Robot command sent successfully")
        
        return action_info

    def create_action_dict(self, action):
        if self.verbose:
            print("RobotEnv.create_action_dict: Creating action dictionary")
        
        result = self._robot.create_action_dict(action)
        
        if self.verbose:
            print("RobotEnv.create_action_dict: Action dictionary created")
        
        return result

    def read_cameras(self):
        if self.verbose:
            print("RobotEnv.read_cameras: Reading camera data")
        
        result = self.camera_reader.read_cameras()
        
        if self.verbose:
            print("RobotEnv.read_cameras: Camera data read successfully")
        
        return result

    def get_state(self):
        if self.verbose:
            print("RobotEnv.get_state: Reading robot state")
        
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        
        if self.verbose:
            print(f"RobotEnv.get_state: Robot state read in {timestamp_dict['read_end'] - timestamp_dict['read_start']}ms")
        
        return state_dict, timestamp_dict

    def get_camera_extrinsics(self, state_dict):
        if self.verbose:
            print("RobotEnv.get_camera_extrinsics: Computing camera extrinsics")
        
        # Adjust gripper camere by current pose
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if hand_camera_id not in cam_id:
                continue
            gripper_pose = state_dict["cartesian_position"]
            extrinsics[cam_id + "_"] = extrinsics[cam_id]
            extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
            
            if self.verbose:
                print(f"RobotEnv.get_camera_extrinsics: Updated extrinsics for camera {cam_id}")
        
        if self.verbose:
            print("RobotEnv.get_camera_extrinsics: Camera extrinsics computation complete")
        
        return extrinsics

    def get_observation(self):
        if self.verbose:
            print("RobotEnv.get_observation: Gathering full observation")
        
        obs_dict = {"timestamp": {}}

        # Robot State #
        if self.verbose:
            print("RobotEnv.get_observation: Reading robot state")
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        if self.verbose:
            print("RobotEnv.get_observation: Reading camera data")
        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        if self.verbose:
            print("RobotEnv.get_observation: Computing camera information")
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        if self.verbose:
            print("RobotEnv.get_observation: Computing camera intrinsics")
        intrinsics = {}
        for cam in self.camera_reader.camera_dict.values():
            cam_intr_info = cam.get_intrinsics()
            for (full_cam_id, info) in cam_intr_info.items():
                intrinsics[full_cam_id] = info["cameraMatrix"]
        obs_dict["camera_intrinsics"] = intrinsics

        if self.verbose:
            print("RobotEnv.get_observation: Observation gathering complete")

        return obs_dict
