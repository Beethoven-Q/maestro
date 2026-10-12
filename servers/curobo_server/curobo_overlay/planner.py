import numpy as np
from curobo.geom.types import WorldConfig, Cuboid
from scipy.spatial.transform import Rotation as R


def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat

# Third Party
import torch
from abc import ABC, abstractmethod

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.camera import CameraObservation


class Planner(ABC):
    def __init__(self, robot_type:str):
        """
            Initialize the planner
        """
        self.robot_type = robot_type

    def setup_scene(self):
        """
            Set up the scene for the planner
        """
        # in x, y, z, qx, qy, qz, qw
        box_name = "desktop"
        # The table top: a large box whose top surface is 2 cm below the robot base
        BOX_DIMS = (4, 4, 1)
        box_pose = np.array([0, 0, -(BOX_DIMS[2]) / 2 - 0.02, 0, 0, 0, 1])
        self.add_object(box_name, box_pose, 'box', size=BOX_DIMS)

    @abstractmethod
    def add_object(self, object_name:str, object_pose:np.ndarray, type:str, **args):
        """  """
        pass

class CuroboPlanner(Planner):
    def __init__(self, robot_type:str):
        """
            Initialize the planner
        """
        super(CuroboPlanner, self).__init__(robot_type)
        self.robot_type = robot_type
        self.world_cfg = WorldConfig.from_dict(
            {
                "blox": {
                    "world": {
                        "pose": [0, 0, 0, 1, 0, 0, 0],
                        "integrator_type": "occupancy",
                        "voxel_size": 0.02,
                    }
                }
            }
        )
        self.setup_scene()

        if self.robot_type == 'franka':
            self.config_file_path =  "franka.yml"
        elif self.robot_type == 'franka_robotiq':
            self.config_file_path = "franka_robotiq.yml"

        # convenience function to store tensor type and device
        self.tensor_args = TensorDeviceType()

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.config_file_path,
            self.world_cfg,
            self.tensor_args,
            trajopt_tsteps=64,
            collision_checker_type=CollisionCheckerType.BLOX,
            use_cuda_graph=False,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.05,
            collision_activation_distance=0.01,
            acceleration_scale=1.0,
            self_collision_check=True,
            maximum_trajectory_dt=0.10,
            finetune_dt_scale=1.05,
            fixed_iters_trajopt=True,
            finetune_trajopt_iters=500,
            minimize_jerk=True,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up..")
        self.motion_gen.warmup(warmup_js_trajopt=False)

    def clean_map(self):
        """ Clean the map """
        self.motion_gen.world_collision.clear_cache()

    def add_object(self, object_name:str, object_pose:np.ndarray, type:str, **args):
        """  """
        if type == 'box':
            size = args['size']
            cube = Cuboid(
                name=object_name,
                pose=[object_pose[0], object_pose[1], object_pose[2], object_pose[6], object_pose[3], object_pose[4], object_pose[5]],
                dims=[size[0], size[1], size[2]],
                color=[0.8, 0.0, 0.0, 1.0],
            )
            self.world_cfg.add_obstacle(cube)

    def plan(self, goal_pose:np.ndarray, start_state:np.ndarray, constrained = False):
        """
            Plan the trajectory

            Args:
                goal_pose: np.ndarray of shape (4, 4) or (7,)
                start_state: np.ndarray of shape (7,)
                constrained: bool, whether to use constrained optimization, the ee will be constrained to the goal pose in z - direction
        """
        goal_cupose = CuroboPlanner.convert_to_pose(goal_pose, self.tensor_args)

        joint_state = JointState(
            position=torch.tensor(start_state[None, :], device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            velocity=torch.zeros((1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            acceleration=torch.zeros((1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
        )
        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=4,
            max_attempts=2,
            enable_finetune_trajopt=True,
            time_dilation_factor=1.0,
        )

        if constrained:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device([1, 1, 1, 1, 1, 0]),
            )
            plan_config.pose_cost_metric = pose_cost_metric

        # Use MotionGen to plan trajectory
        result = self.motion_gen.plan_single(joint_state, goal_cupose, plan_config)

        if not result.success:
            print(result.status)
            return False, None, None

        # Use optimized plan and dt for time-optimal execution
        traj = result.optimized_plan  # Optimized waypoints (fewer, more efficient)
        dt = result.optimized_dt      # Scalar dt - same between all waypoints

        # Convert scalar tensor to float for easier use
        if hasattr(dt, 'item'):
            dt = dt.item()  # Convert tensor to Python float

        return result.success, traj, dt

    def plan_grasp(self, goal_pose:np.ndarray, start_state:np.ndarray):
        """
            Plan an approach to the best reachable of several grasp poses

            Args:
                goal_pose: np.ndarray of shape (N, 4, 4) or (N, 7)
                start_state: np.ndarray of shape (7,)
        """
        goal_cupose = CuroboPlanner.convert_to_pose(goal_pose, self.tensor_args, batchify=True)

        joint_state = JointState(
            position=torch.tensor(start_state[None, :], device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            velocity=torch.zeros((1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            acceleration=torch.zeros((1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
        )

        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=4,
            max_attempts=10,
            enable_finetune_trajopt=True,
            time_dilation_factor=1.0,
        )

        # Use MotionGen to plan trajectory
        result = self.motion_gen.plan_grasp(joint_state, goal_cupose,
                                                plan_config)

        if not result.success:
            print(result.status)
            return False, None

        # A GraspPlanResult: the server returns its grasp_interpolated_trajectory and grasp_interpolation_dt
        return result.success, result

    def fuse_observation(self, dataset, mask_gripper=True, gripper_depth_range=(0.0, 0.25)):
        """Fuse depth views into the nvBlox world model.

        Args:
            dataset: views, each a dict with 'depths' (H, W meters), 'K' (3x3) and 'camtoworld' (4x4)
            mask_gripper: If True, filter out points within gripper_depth_range (wrist camera)
            gripper_depth_range: (min, max) depth in meters to filter (default: 0-25cm for wrist-mounted camera)
        """
        world_model = self.motion_gen.world_collision
        print(f"[curobo] fuse_observation: fusing {len(dataset)} views "
              f"(mask_gripper={mask_gripper}, gripper_range={gripper_depth_range})")
        for data_pack in dataset:
            depth = np.asarray(data_pack['depths'])
            cam_K = np.asarray(data_pack['K'])
            c2w = np.asarray(data_pack['camtoworld'])
            if depth.ndim != 2:
                raise ValueError("Depth must be a 2D map")

            # Mask out gripper (for wrist-mounted cameras)
            if mask_gripper:
                depth_min, depth_max = gripper_depth_range
                depth = depth.copy()
                depth[(depth >= depth_min) & (depth <= depth_max)] = 0.0  # Invalidate gripper depths

            position = torch.from_numpy(c2w[:3, 3]).float()
            orientation = torch.from_numpy(rmat_to_quat(c2w[:3, :3])[[3, 0, 1, 2]]).float()

            camera_pose = Pose(
                position=self.tensor_args.to_device(position).float(),
                quaternion=self.tensor_args.to_device(orientation).float(),
            )

            data_camera = CameraObservation(
                depth_image=torch.from_numpy(depth).float(),
                intrinsics=torch.from_numpy(cam_K).float(),
                pose=camera_pose,
            )
            data_camera = data_camera.to(device=self.tensor_args.device)
            world_model.add_camera_frame(data_camera, "world")
            world_model.process_camera_frames("world", False)
            torch.cuda.synchronize()
            world_model.update_blox_hashes()

    def toggle_collision_for_grasp(self, flag = False):
        if self.robot_type == 'franka':
            self.motion_gen.toggle_link_collision(["panda_leftfinger", "panda_rightfinger"], flag)
        elif self.robot_type == 'franka_robotiq':
            self.motion_gen.toggle_link_collision(
                ["robotiq_2f_85_right_driver",
                "robotiq_2f_85_right_follower",
                "robotiq_2f_85_left_follower",
                "robotiq_2f_85_right_coupler",
                "robotiq_2f_85_right_pad",
                "robotiq_2f_85_right_spring_link",
                "robotiq_2f_85_left_driver",
                "robotiq_2f_85_left_follower",
                "robotiq_2f_85_left_coupler",
                "robotiq_2f_85_left_pad",
                "robotiq_2f_85_left_spring_link",], flag)

    @staticmethod
    def convert_to_pose(poses:np.ndarray, tensor_args:TensorDeviceType = None, batchify = False):
        if tensor_args is None:
            tensor_args = TensorDeviceType()

        ee_position = []
        ee_quaternion = [] # w, x, y, z

        if poses.ndim >= 2 and poses.shape[-1] == 4 and poses.shape[-2] == 4:
            poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                ee_position.append(pose[:3, 3])
                ee_quaternion.append(rmat_to_quat(pose[:3, :3])[[3, 0, 1, 2]])

        elif poses.ndim >= 2 and poses.shape[-1] == 7:
            poses = poses.reshape(-1, 7)
            for pose in poses:
                ee_position.append(pose[:3])
                ee_quaternion.append(pose[3:])

        elif poses.ndim == 1 and poses.shape[0] == 7:
            ee_position.append(poses[:3])
            ee_quaternion.append(poses[3:])

        ee_position = torch.tensor(np.array(ee_position), device=tensor_args.device, dtype=tensor_args.dtype)
        ee_quaternion = torch.tensor(np.array(ee_quaternion), device=tensor_args.device, dtype=tensor_args.dtype)

        if batchify:
            return Pose(ee_position.view(1, -1, 3), ee_quaternion.view(1, -1, 4), batch=ee_position.shape[0])
        else:
            if len(ee_position) == 1:
                return Pose(ee_position, ee_quaternion, batch=ee_position.shape[0])
            else:
                return Pose(ee_position.view(-1, 1, 3), ee_quaternion.view(-1, 1, 4), batch=ee_position.shape[0])
