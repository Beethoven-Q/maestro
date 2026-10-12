import numpy as np
import cv2

import PIL.Image as Image
import os
from droid.robot_env import RobotEnv
from droid.misc.transformations import euler_to_rmat, rmat_to_euler
from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
from maestro.config import cfg, service_url
import json
from droid.data_processing.timestep_processing import TimestepProcesser
import time
from droid.misc.transformations import pose_diff
import threading
from concurrent.futures import ThreadPoolExecutor
from maestro.perception_utils import (fuse_observation, gripper_mask, stereo_depth, segment,
                                      draw_segmentation, sample_grasps)
from maestro.camera_geometry import camera_matrix, stereo_geometry
from maestro import vlm
import requests
from datetime import datetime

SN = hand_camera_id

# ----------------------------------------
# Scene scan cache for reusing wrist scans
# ----------------------------------------
class SceneScanCache:
    """Cache wrist-camera scene scans to avoid repeated rescans.

    Stores for each candidate wrist pose:
    - target_joints: absolute 7-DoF joint configuration used to capture images
    - left_image, right_image: RGB numpy arrays captured at that pose
    - timestamp: capture time
    The cache is invalidated by refresh_scan, or when a target cannot be found or verified in the cached views.
    """
    def __init__(self, base_dir: str):
        self.entries = []
        self.valid = False
        self.base_dir = base_dir

    def set(self, entries):
        self.entries = entries
        self.valid = True

    def get(self):
        return self.entries if self.valid else []

    def invalidate(self):
        self.entries = []
        self.valid = False

    def _ensure_dir(self):
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception:
            pass

    def save_to_disk(self, robot_api=None):
        """Persist current cache entries to disk under base_dir and write meta.json.
        
        Args:
            robot_api: Optional RealFrankaRobotApi instance for dual-save and logging
        """
        if not self.valid or len(self.entries) == 0:
            return
        self._ensure_dir()
        
        # Generate timestamp for this save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Clear existing files in base_dir (latest_data)
        try:
            for name in os.listdir(self.base_dir):
                try:
                    os.remove(os.path.join(self.base_dir, name))
                except Exception:
                    pass
        except Exception:
            pass

        meta = []
        for i, entry in enumerate(self.entries, start=1):
            # Save to latest_data (no timestamp)
            left_path = os.path.join(self.base_dir, f"pose_{i}_left.png")
            right_path = os.path.join(self.base_dir, f"pose_{i}_right.png")
            depth_path_npy = os.path.join(self.base_dir, f"pose_{i}_depth.npy")
            
            try:
                Image.fromarray(entry['left_image']).save(left_path)
                Image.fromarray(entry['right_image']).save(right_path)
                # If depth is available, save it too
                if 'depth' in entry and isinstance(entry['depth'], np.ndarray):
                    try:
                        np.save(depth_path_npy, entry['depth'])
                    except Exception:
                        pass
                # If robot_api is available, also save timestamped versions via dual-save
                if robot_api:
                    robot_api._save_dual(entry['left_image'], f"scan_cache/pose_{i}_left", "png", 
                                       timestamp, context="scan_workspace")
                    robot_api._save_dual(entry['right_image'], f"scan_cache/pose_{i}_right", "png",
                                       timestamp, context="scan_workspace")
                    if 'depth' in entry and isinstance(entry['depth'], np.ndarray):
                        robot_api._save_dual(entry['depth'], f"scan_cache/pose_{i}_depth", "npy",
                                           timestamp, is_image=False, context="scan_workspace")
                    robot_api._log_timestamp("FILE_SAVE", f"scan_cache pose {i} images saved")
                    robot_api._log_timestamp("FILE_SAVE", f"scan_cache pose {i} depth saved")
            except Exception:
                # Skip if saving fails for this entry
                continue
                
            meta_entry = {
                'pose_index': i,
                'target_joints': np.array(entry['target_joints']).tolist(),
                'timestamp': float(entry.get('timestamp', time.time())),
                'left_image': os.path.basename(left_path),
                'right_image': os.path.basename(right_path),
            }
            if os.path.exists(depth_path_npy):
                meta_entry['depth_file'] = os.path.basename(depth_path_npy)
            # Persist intrinsics and camera pose if present
            if 'intrinsics' in entry:
                try:
                    meta_entry['intrinsics'] = camera_matrix(entry['intrinsics']).tolist()
                except Exception:
                    pass
            if 'cam_pose' in entry:
                try:
                    meta_entry['cam_pose'] = np.array(entry['cam_pose']).tolist()
                except Exception:
                    pass
            meta.append(meta_entry)
            
        # Save metadata
        try:
            meta_path = os.path.join(self.base_dir, 'meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
                
            # Save timestamped metadata if robot_api available
            if robot_api:
                robot_api._save_dual(meta, "scan_cache/meta", "json", 
                                   timestamp, is_image=False, context="scan_workspace")
                robot_api._log_timestamp("FILE_SAVE", f"scan_cache metadata saved with {len(meta)} poses")
        except Exception:
            pass

    def load_from_disk(self):
        """Load cached entries from disk if available."""
        self._ensure_dir()
        meta_path = os.path.join(self.base_dir, 'meta.json')
        if not os.path.exists(meta_path):
            return []
        try:
            meta = json.load(open(meta_path, 'r'))
        except Exception:
            return []
        loaded = []
        for m in meta:
            left_path = os.path.join(self.base_dir, m.get('left_image', ''))
            right_path = os.path.join(self.base_dir, m.get('right_image', ''))
            if not (os.path.exists(left_path) and os.path.exists(right_path)):
                continue
            try:
                left_img = np.array(Image.open(left_path))
                right_img = np.array(Image.open(right_path))
                depth_file = m.get('depth_file')
                loaded.append({
                    'target_joints': np.array(m.get('target_joints', []), dtype=float),
                    'left_image': left_img,
                    'right_image': right_img,
                    'timestamp': float(m.get('timestamp', time.time())),
                    'depth': np.load(os.path.join(self.base_dir, depth_file)) if depth_file else None,
                    'intrinsics': m.get('intrinsics'),
                    'cam_pose': m.get('cam_pose'),
                })
            except Exception:
                continue
        return loaded

    def ensure_loaded_from_disk(self):
        if not self.valid:
            entries = self.load_from_disk()
            if len(entries) > 0:
                self.set(entries)

GRIPPER_FLANGE_OFFSET = np.array([-0.00, -0.00, 0.14])

REACH_TOLERANCE = 0.05        # joint-space distance (rad) at which a target counts as reached
MAX_REACH_CORRECTION = 0.1    # largest per-joint aim-past correction (rad); larger errors are retried as is
LOOK_AROUND_DISTANCE = 0.32   # camera-to-object distance (m) for the look-around views before grasping
# Look-around views as (degrees to the side, degrees lower), tried in order.
LOOK_AROUND_VIEWS = [(0, 0), (35, 0), (-35, 0), (0, 15), (35, 15), (-35, 15)]

WRIST_DEPTH_FRAMES = 3        # fresh frames to try before falling back to the object's depth
MIN_MASK_DEPTH_FRACTION = 0.1 # share of the object mask that must have valid depth for the fallback
VIDEO_CAMERA_ID = varied_camera_2_id or varied_camera_1_id  # external camera the trial video records


class RealFrankaRobotApi():
    """The robot tools the planner's code calls as `robot.<tool>(...)`."""
    def __init__(self, env: RobotEnv, instruction, use_motionplanner=True):
        self.env = env
        self.instruction = instruction
        self.timeprocesser = TimestepProcesser(ignore_action=True)

        # Video recording with direct-to-disk writing for memory efficiency
        self.video_writer = None  # OpenCV VideoWriter instance
        self.video_path = None  # Path to output video file
        self.video_frame_count = 0  # Track number of frames recorded
        self._recorder_thread = None  # Background thread for continuous recording
        self._recorder_stop_evt = threading.Event()  # Signal to stop recording
        self._video_lock = threading.Lock()  # Thread safety for video writer
        self._video_active = False  # Flag to prevent double saves
        self._video_width = None  # Actual width from camera
        self._video_height = None  # Actual height from camera

        extrinsics = self.env.get_observation()['camera_extrinsics']
        self.camera_to_world_matrix = []
        for serial in (varied_camera_1_id, varied_camera_2_id, hand_camera_id):
            pose = extrinsics.get(f'{serial}_left') if serial else None
            if pose is None:
                if serial == hand_camera_id:
                    raise ValueError(f'Missing wrist calibration for {serial}_left')
                self.camera_to_world_matrix.append(None)
                continue
            matrix = np.eye(4)
            matrix[:3, :3] = euler_to_rmat(pose[3:6])
            matrix[:3, 3] = pose[:3]
            self.camera_to_world_matrix.append(matrix)
        print("Camera to World Matrices:", self.camera_to_world_matrix)

        # Intrinsics with a fallback if a camera can't be read
        try:
            params = self.env.camera_reader.get_camera(varied_camera_1_id)._cam.get_camera_information().camera_configuration.calibration_parameters.right_cam
        except Exception:
            params = type('obj', (object,), {'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0})
        intrinsics_matrix_1 = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])
        try:
            params = self.env.camera_reader.get_camera(varied_camera_2_id)._cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        except Exception:
            params = type('obj', (object,), {'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0})
        intrinsics_matrix_2 = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])
        try:
            params = self.env.camera_reader.get_camera(hand_camera_id)._cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        except Exception:
            params = type('obj', (object,), {'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0})
        intrinsics_matrix_3 = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])
        self.intrinsics_matrix = [intrinsics_matrix_1, intrinsics_matrix_2, intrinsics_matrix_3]
        self.use_motionplanner = use_motionplanner
        self.image_editor = ImageEditApi(env)

        self.logger = None
        requests.post(f"{service_url('curobo')}/clean_map")
        # Wrist scan cache - will be updated with trial directory later
        self.scan_cache = SceneScanCache(os.path.join(cfg.paths['data_dir'], "grasp_data", "scan_cache"))
        self.verify_logit_drop_ratio = 0.7
        self.verify_abs_min_logit = 0.50
        self.is_scanning_wrist = False

    def _log_timestamp(self, event_type, details):
        """Log timestamp for function calls or file saves"""
        if hasattr(self.logger, 'log_dir'):
            timestamp_file = self.logger.log_dir / "timestamps.log"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_entry = f"{timestamp} | {event_type} | {details}\n"
            try:
                with open(timestamp_file, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"Failed to write timestamp log: {e}")
    
    def _update_manifest(self, filename_base, extension, timestamp, context=None):
        """Update the manifest file that tracks file mappings between latest and visualization."""
        try:
            latest_dir, _ = self._get_trial_dirs()
            manifest_path = os.path.join(os.path.dirname(latest_dir), "manifest.json")
            
            # Load existing manifest or create new
            manifest = {}
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            
            # Add new entry
            key = f"{filename_base}.{extension}"
            if key not in manifest:
                manifest[key] = []
            
            manifest[key].append({
                "timestamp": timestamp,
                "viz_file": f"{filename_base}_{timestamp}.{extension}",
                "context": context,
                "real_time": datetime.now().isoformat()
            })
            
            # Save updated manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
        except Exception as e:
            print(f"Failed to update manifest: {e}")

    def _get_trial_dirs(self):
        """Get both latest_data and visualization directories for the current trial."""
        base = os.path.join(cfg.paths['data_dir'], "grasp_data")
        try:
            trial_name = self.logger.log_dir.name if (hasattr(self, 'logger') and self.logger) else None
        except Exception:
            trial_name = None
        
        if trial_name:
            trial_base = os.path.join(base, trial_name)
            latest_dir = os.path.join(trial_base, "latest_data")
            viz_dir = os.path.join(trial_base, "visualization")
        else:
            # Fallback to root grasp_data if no trial name
            latest_dir = base
            viz_dir = base
        
        # Ensure directories exist
        try:
            os.makedirs(latest_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)
        except Exception:
            pass
        
        return latest_dir, viz_dir
    
    def _save_dual(self, data, filename_base, extension, timestamp=None, is_image=True, context=None):
        """Save file in both latest_data (without timestamp) and visualization (with timestamp) folders.
        
        Args:
            data: The data to save (numpy array for images, dict/other for json, etc.)
            filename_base: Base filename without extension (e.g., "rgb_3", "crop_img")
            extension: File extension (e.g., "png", "json", "npy")
            timestamp: Optional timestamp string, will generate if not provided
            is_image: Whether to save as image using PIL
            context: Optional context string for the manifest (e.g., "segmentation", "grasp_planning")
        
        Returns:
            Tuple of (latest_path, viz_path) for the saved files
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        latest_dir, viz_dir = self._get_trial_dirs()
        
        # Handle subdirectories in filename_base (e.g., "scan_cache/pose_1_left")
        if '/' in filename_base:
            subdir = os.path.dirname(filename_base)
            base_name = os.path.basename(filename_base)
            # Create subdirectories if they don't exist
            latest_subdir = os.path.join(latest_dir, subdir)
            viz_subdir = os.path.join(viz_dir, subdir)
            os.makedirs(latest_subdir, exist_ok=True)
            os.makedirs(viz_subdir, exist_ok=True)
            # Save to latest_data without timestamp (for active use)
            latest_path = os.path.join(latest_subdir, f"{base_name}.{extension}")
            # Save to visualization with timestamp (for post-processing)
            viz_path = os.path.join(viz_subdir, f"{base_name}_{timestamp}.{extension}")
        else:
            # Save to latest_data without timestamp (for active use)
            latest_path = os.path.join(latest_dir, f"{filename_base}.{extension}")
            # Save to visualization with timestamp (for post-processing)
            viz_path = os.path.join(viz_dir, f"{filename_base}_{timestamp}.{extension}")
        
        try:
            if is_image:
                # Save as image
                if isinstance(data, np.ndarray):
                    img = Image.fromarray(data)
                else:
                    img = data
                img.save(latest_path)
                img.save(viz_path)
            elif extension == "json":
                # Save as JSON
                with open(latest_path, 'w') as f:
                    json.dump(data, f)
                with open(viz_path, 'w') as f:
                    json.dump(data, f)
            elif extension == "npy":
                # Save as numpy array
                np.save(latest_path, data)
                np.save(viz_path, data)
            else:
                # Generic file save (for other types)
                with open(latest_path, 'wb') as f:
                    f.write(data)
                with open(viz_path, 'wb') as f:
                    f.write(data)
            
            # Log to timestamp file with mapping
            self._log_timestamp("FILE_SAVE", f"{filename_base}.{extension} -> {filename_base}_{timestamp}.{extension}")
            
            # Update manifest with file mapping
            self._update_manifest(filename_base, extension, timestamp, context)
            
        except Exception as e:
            print(f"Error saving dual files: {e}")
        
        return latest_path, viz_path
    
    def _load_latest(self, filename):
        """Load file from latest_data directory."""
        latest_dir, _ = self._get_trial_dirs()
        path = os.path.join(latest_dir, filename)
        
        if os.path.exists(path):
            self._log_timestamp("FILE_LOAD", f"Loaded from: {path}")
        
        return path

    def _get_processed_observation(self):
        """Get processed observation from the environment."""
        obs = self.env.get_observation()
        timestep = {'observation': obs}
        timestep = self.timeprocesser.forward(timestep)
        return timestep

    def generate_best_grid_from_mask(self, mask, target_range=(25, 35), boundary_sample_num=10):
        ys, xs = np.where(mask == 1)
        if len(xs) == 0:
            raise ValueError("Mask contains no valid points.")
        boundary = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if len(boundary) == 0:
            raise ValueError("No boundary found.")
        boundary_pts = boundary[0].squeeze(1)
        boundary_len = len(boundary_pts)
        indices = np.linspace(0, boundary_len - 1, boundary_sample_num).astype(int)
        sampled_boundary_pts = boundary_pts[indices]
        sampled_boundary_pts = np.flip(sampled_boundary_pts, axis=1)
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        best_internal_pts = None
        for stride in range(min(height, width), 1, -1):
            grid_x = np.arange(min_x +5, max_x - 4, stride)
            grid_y = np.arange(min_y + 5, max_y - 4, stride)
            grid = np.array(np.meshgrid(grid_y, grid_x)).reshape(2, -1).T
            valid_internal_pts = [pt for pt in grid if mask[pt[0], pt[1]] == 1]
            if target_range[0] <= len(valid_internal_pts) <= target_range[1]:
                best_internal_pts = np.array(valid_internal_pts)
                break
            if best_internal_pts is None or len(valid_internal_pts) > len(best_internal_pts):
                best_internal_pts = np.array(valid_internal_pts)
        if best_internal_pts is None:
            all_pts = sampled_boundary_pts
        else:
            best_internal_pts = np.array(best_internal_pts).reshape(-1, 2)
            all_pts = np.concatenate([sampled_boundary_pts, best_internal_pts], axis=0)
        return all_pts

    def _segment(self, image, object_name, camera_id=2):
        """GroundedSAM detections of `object_name` in an RGB wrist image.

        Returns the mask.json entries (background first, then one entry per detection with
        its label, score as "logit" and pixel box) and the detection masks. Both are saved
        with the trial as mask.json and masks.npy, next to the input and a drawn overlay.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._save_dual(image, f"rgb_{camera_id+1}", "png", timestamp, context="segmentation_input")
        detections = segment(image, object_name)
        mask_json = [{"value": 0, "label": "background"}] + [
            {"value": i, "label": str(label), "logit": float(score), "box": box.tolist()}
            for i, (label, score, box) in enumerate(
                zip(detections['labels'], detections['scores'], detections['boxes']), start=1)]
        self._save_dual(detections['masks'][:, None], "masks", "npy", timestamp, is_image=False, context="segmentation")
        self._save_dual(mask_json, "mask", "json", timestamp, is_image=False, context="segmentation")
        self._save_dual(draw_segmentation(image, detections), f"segmentation_{camera_id+1}", "jpg", timestamp,
                        context="segmentation")
        self._log_timestamp("SEGMENTATION", f"'{object_name}': {len(mask_json) - 1} detection(s)")
        return mask_json, detections['masks']

    def _segment_current_view(self, object_name, camera_id=2):
        try:
            obs = self._get_processed_observation()
            image = obs['observation']['camera']['image']['hand_camera'][0][..., :3][..., ::-1]
            mask_json, masks = self._segment(image, object_name, camera_id)
            if len(mask_json) < 2:
                return None, None, None
            mask = masks[0]
            data = self.env.get_observation()
            ext = np.array(data['camera_extrinsics'][f'{SN}_left'])
            ext_mat = np.eye(4)
            ext_mat[:3, :3] = euler_to_rmat(ext[3:6])
            ext_mat[:3, 3] = ext[:3]
            return mask.astype(np.uint8), ext_mat, image
        except Exception:
            return None, None, None

    def _estimate_target_centroid_world(self, object_name, camera_id=2):
        mask, ext_mat, _ = self._segment_current_view(object_name, camera_id=camera_id)
        if mask is None:
            return None
        xyz_map = self._get_pointcloud()
        pts = xyz_map[mask == 1]
        if pts.shape[0] == 0:
            return None
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
        pts = pts[valid]
        if pts.shape[0] == 0:
            return None
        hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        pts_w = (hom @ ext_mat.T)[:, :3]
        centroid = np.median(pts_w, axis=0)
        return centroid

    def _get_ee_to_cam_transform(self):
        ee_pose = self.env._robot.get_ee_pose()
        ee_T = np.eye(4)
        ee_T[:3, :3] = euler_to_rmat(ee_pose[3:])
        ee_T[:3, 3] = ee_pose[:3]
        data = self.env.get_observation()
        ext = np.array(data['camera_extrinsics'][f'{SN}_left'])
        cam_T = np.eye(4)
        cam_T[:3, :3] = euler_to_rmat(ext[3:6])
        cam_T[:3, 3] = ext[:3]
        ee_to_cam = np.linalg.inv(ee_T) @ cam_T
        return ee_to_cam

    def _look_at_camera_pose(self, cam_pos_w, target_w):
        cam_pos_w = np.asarray(cam_pos_w, dtype=float)
        target_w = np.asarray(target_w, dtype=float)
        f = target_w - cam_pos_w
        f_norm = np.linalg.norm(f)
        if f_norm < 1e-6:
            f = np.array([0.0, 0.0, 1.0])
        else:
            f = f / f_norm
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(f, up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0])
        r = np.cross(up, f)
        r = r / (np.linalg.norm(r) + 1e-8)
        u = np.cross(f, r)
        Rc = np.eye(3)
        Rc[:, 0] = r
        Rc[:, 1] = u
        Rc[:, 2] = f
        Tcw = np.eye(4)
        Tcw[:3, :3] = Rc
        Tcw[:3, 3] = cam_pos_w
        return Tcw

    def _move_camera_to(self, cam_pos_w, target_w):
        """Point the wrist camera from cam_pos_w at target_w; True if the arm moved there."""
        T_ee_to_cam = self._get_ee_to_cam_transform()
        T_cam = self._look_at_camera_pose(cam_pos_w, target_w)
        T_ee = T_cam @ np.linalg.inv(T_ee_to_cam)
        fingertip = T_ee[:3, 3] + T_ee[:3, :3] @ GRIPPER_FLANGE_OFFSET
        return self.move_to_pose(fingertip, rmat_to_euler(T_ee[:3, :3]))

    def _plan_look_around_positions(self, target_world):
        """Candidate camera positions around the target, most preferred first.

        The first three look at the target from the current viewing direction and 35
        degrees to either side. The rest are fallbacks for when shelves or other structure
        make some of those unreachable: the same directions 15 degrees lower, then 60
        degrees to either side.
        """
        robot_state = self.env._robot.get_robot_state()
        pose = robot_state[0]['cartesian_position']
        gripper_center = np.asarray(pose[:3]) + euler_to_rmat(pose[3:]) @ GRIPPER_FLANGE_OFFSET
        view = np.asarray(target_world, dtype=float) - gripper_center
        view /= np.linalg.norm(view) + 1e-8
        side = np.cross(view, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) < 1e-3:
            side = np.cross(view, [0.0, 1.0, 0.0])
        side /= np.linalg.norm(side)
        down = np.cross(view, side)
        if down[2] > 0:
            down = -down
        positions = []
        for yaw_deg, lower_deg in LOOK_AROUND_VIEWS:
            yaw, lower = np.deg2rad(yaw_deg), np.deg2rad(lower_deg)
            direction = np.cos(yaw) * view + np.sin(yaw) * side
            direction = np.cos(lower) * direction - np.sin(lower) * down
            positions.append(np.asarray(target_world) - LOOK_AROUND_DISTANCE * direction / np.linalg.norm(direction))
        return positions

    def _collect_points_at_current_view(self, object_name, camera_id=2, max_points=8000):
        mask, ext_mat, color_img = self._segment_current_view(object_name, camera_id=camera_id)
        if mask is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        xyz_map = self._get_pointcloud()
        color_full = color_img
        pts = xyz_map[mask == 1]
        cols = color_full[mask == 1]
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
        pts = pts[valid]
        cols = cols[valid]
        if pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
            cols = cols[idx]
        hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        pts_w = (hom @ ext_mat.T)[:, :3]
        return pts_w.astype(np.float32), cols.astype(np.uint8)

    def _scan_workspace_store_cache(self):
        """Scan multiple wrist poses and cache the images for later use."""
        self._log_timestamp("SUBTASK_START", "Wrist camera scan for scene observation")
        
        reset_pose = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
        poses = [
            np.array([0 ,0 ,0 ,0 ,0 , 0, 0]),
            np.array([np.pi/5 ,0 ,0 ,0 ,0 ,np.pi/10, 0]),
            np.array([np.pi/5 ,0 ,0 ,np.pi/10 ,0 ,np.pi/10, 0]),
            np.array([-np.pi/5 ,0 ,0 ,0 ,0 ,np.pi/10, 0]),
            np.array([-np.pi/5 ,0 ,0 ,np.pi/10 ,0 ,np.pi/10, 0]),
            np.array([0 ,0 ,0 , np.pi/10, 0 ,np.pi/10, 0]),
            np.array([0 ,0 ,0 , np.pi/5, 0 ,np.pi/10, 0]),
            np.array([0 ,0 ,0 ,0 ,0 , 0, 0]),

        ]
        calibration = self.env.camera_reader.get_camera(str(hand_camera_id)).get_calibration()
        cache_entries = []
        depth_jobs = []
        # Stereo depth for the collision map is computed while the arm moves to the next pose.
        with ThreadPoolExecutor(max_workers=1) as depth_worker:
            for i, pose in enumerate(poses, 1):
                target_joints = reset_pose + pose
                self._log_timestamp("SCAN_POSE", f"Moving to scan pose {i}/{len(poses)}")
                # Capture even if a pose is slightly off: images are placed with the measured pose.
                self._hold_and_wait_for_reach(target_joints)
                time.sleep(0.3)
                obs = self._get_processed_observation()
                images = obs['observation']['camera']['image']['hand_camera']
                left_img = images[0][..., :3][..., ::-1].copy()
                right_img = images[1][..., :3][..., ::-1].copy()
                depth_jobs.append(depth_worker.submit(stereo_depth, left_img, right_img, calibration))
                intr = None
                try:
                    intr = camera_matrix(obs["intrinsics_dict"]["hand_camera"][0])
                except Exception:
                    try:
                        intr = self.intrinsics_matrix[2]
                    except Exception:
                        intr = None
                cam_pose = None
                try:
                    ext = obs['extrinsics_dict']['hand_camera'][0]
                    cam_pose = np.zeros((4,4), dtype=np.float32)
                    cam_pose[:3,:3] = euler_to_rmat(ext[3:6])
                    cam_pose[:3,3] = ext[:3]
                    cam_pose[3,3] = 1.0
                except Exception:
                    cam_pose = None
                cache_entries.append({
                    'target_joints': np.array(target_joints, dtype=float),
                    'left_image': left_img,
                    'right_image': right_img,
                    'timestamp': time.time(),
                    'depth': None,
                    'intrinsics': intr,
                    'cam_pose': cam_pose,
                })
                self._log_timestamp("SCAN_CAPTURE", f"Captured images for pose {i}")
            depths = [job.result() for job in depth_jobs]
        # The gripper's depth is unreliable at this range; keep it out of the collision map.
        if depths:
            gripper = gripper_mask(depths)
            for entry, depth in zip(cache_entries, depths):
                depth[gripper] = 0.0
                entry['depth'] = depth

        self.scan_cache.set(cache_entries)
        self._log_timestamp("SUBTASK_END", f"Wrist scan completed with {len(cache_entries)} poses")
        
        try:
            self.scan_cache.save_to_disk(robot_api=self)
        except Exception:
            pass

    def _segment_on_images(self, object_name, camera_id, image):
        try:
            mask_json, _ = self._segment(image, object_name, camera_id)
            return mask_json
        except Exception as e:
            self._log_timestamp("SEGMENTATION_ERROR", f"Segmentation error for '{object_name}': {e}")
            return None

    def _select_best_from_cache(self, object_name, camera_id=2):
        entries = self.scan_cache.get()
        best_idx = -1
        best_logit = -1.0
        print(f"[cache] Evaluating {len(entries)} cached wrist views for '{object_name}'...")
        for i, entry in enumerate(entries):
            mask_json = self._segment_on_images(object_name, camera_id, entry['left_image'])
            if mask_json is None or len(mask_json) < 2:
                print(f"[cache] View {i+1}: invalid segmentation result")
                continue
            
            # Handle both 'logit' and 'box' fields for compatibility
            logit = -1.0
            if 'logit' in mask_json[1]:
                try:
                    logit = float(mask_json[1]['logit'])
                except Exception:
                    logit = -1.0
            elif 'box' in mask_json[1]:
                # Server may not return logit, use default if box exists
                logit = 0.5
            print(f"[cache] View {i+1}: logit={logit:.3f}")
            if logit > best_logit:
                best_logit = logit
                best_idx = i
        if best_idx >= 0:
            print(f"[cache] Best cached view index={best_idx+1}, logit={best_logit:.3f}")
        else:
            print(f"[cache] No valid cached view found for '{object_name}'")
        return best_idx, best_logit

    def _verify_current_view(self, object_name, camera_id=2):
        obs = self._get_processed_observation()
        live_left = obs['observation']['camera']['image']['hand_camera'][0][..., :3][..., ::-1]
        mask_json = self._segment_on_images(object_name, camera_id, live_left)
        logit = -1.0
        if mask_json is not None and len(mask_json) >= 2:
            if 'logit' in mask_json[1]:
                try:
                    logit = float(mask_json[1]['logit'])
                except Exception:
                    logit = -1.0
            elif 'box' in mask_json[1]:
                # Server may not return logit, use default if box exists
                logit = 0.5
        return mask_json, logit

    def detect_bounding_box(self, object_name):
        """Get a quick bounding box for an object using the current wrist view.

        Returns:
            dict | None: {
                'box': [x1, y1, x2, y2],
                'logit': float
            } or None if unavailable.
        """
        self._log_timestamp("FUNCTION_CALL", f"detect_bounding_box(object_name='{object_name}')")
        mask_json, logit = self._verify_current_view(object_name)
        if mask_json is None or len(mask_json) < 2 or 'box' not in mask_json[1]:
            return None
        try:
            box = mask_json[1]['box']
            # Ensure list of 4 numbers
            if isinstance(box, (list, tuple)) and len(box) == 4:
                box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                return {"box": box, "logit": float(logit)}
        except Exception:
            pass
        return None

    def locate_object_by_segmentation(self, object_name, z_offset=0.0):
        """Coarse 3D interaction point via bounding-box center using pre-scan cache.

        Pipeline (consistent with precise pointing stack):
        _select_best_from_cache -> _ensure_target_visible -> detect_bounding_box -> 3D projection

        Args:
            object_name: Text prompt for grounded segmentation (e.g., "red cup")
            z_offset: Optional vertical offset in meters applied in world Z

        Returns:
            dict | None: {
                'box': [x1, y1, x2, y2],
                'pixel': [y, x],
                '3d_position': [xw, yw, zw],
                'logit': float
            } or None if failed.
        """
        self._log_timestamp("FUNCTION_CALL", f"locate_object_by_segmentation(object_name='{object_name}', z_offset={z_offset})")

        # Ensure we are at a verified good view using the scan cache, like precise pipeline
        verify_json = self._ensure_target_visible(object_name)
        if verify_json is None:
            return None

        # Now perform a lightweight bbox read from this verified view
        bbox_info = self.detect_bounding_box(object_name)
        if bbox_info is None:
            print(f"'{object_name}' was not detected in the live wrist view.")
            return None
        box = bbox_info['box']

        # Center pixel of the bbox; the mask comes from the same segmentation call
        cx = int(round((box[0] + box[2]) / 2.0))
        cy = int(round((box[1] + box[3]) / 2.0))
        mask = np.load(self._load_latest("masks.npy"))[0][0]
        xyz_world = self._lift_pixel_to_world((cy, cx), mask=mask)
        if xyz_world is None:
            print(f"'{object_name}' was detected, but the wrist camera has no valid depth "
                  f"at the box center or on enough of the object.")
            return None
        xyz_world[2] += float(z_offset)

        return {
            'box': box,
            'pixel': [int(cy), int(cx)],
            '3d_position': [float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])],
            'logit': float(bbox_info.get('logit', 0.0))
        }

    def _ensure_target_visible(self, object_name, camera_id=2):
        if not self.scan_cache.valid:
            self.scan_cache.ensure_loaded_from_disk()
        if not self.scan_cache.valid:
            self._scan_workspace_store_cache()
            self.scan_cache.save_to_disk(robot_api=self)
        best_idx, best_logit = self._select_best_from_cache(object_name, camera_id=camera_id)
        if best_idx < 0:
            print("[ensure] No match in cache, rescanning...")
            self.scan_cache.invalidate()
            self._scan_workspace_store_cache()
            self.scan_cache.save_to_disk(robot_api=self)
            best_idx, best_logit = self._select_best_from_cache(object_name, camera_id=camera_id)
            if best_idx < 0:
                print("[ensure] No view found after rescan.")
                return None
        entries = self.scan_cache.get()
        target_joints = entries[best_idx]['target_joints']
        print(f"[verify] Moving to cached pose index={best_idx+1} to verify visibility...")
        self._hold_and_wait_for_reach(target_joints)
        time.sleep(0.3)
        verify_json, verify_logit = self._verify_current_view(object_name, camera_id=camera_id)
        ratio = (verify_logit / best_logit) if best_logit > 0 else 0.0
        print(f"[verify] Cached best logit={best_logit:.3f}, live logit={verify_logit:.3f}, ratio={ratio:.3f}")
        if (
            verify_json is not None and
            len(verify_json) >= 2 and 'box' in verify_json[1] and
            verify_logit >= max(self.verify_abs_min_logit, self.verify_logit_drop_ratio * best_logit)
        ):
            print("[verify] Target visible at cached pose. Proceeding.")
            return verify_json
        print("[ensure] Live verification too weak vs cached best. Rescanning and retrying...")
        self.scan_cache.invalidate()
        self._scan_workspace_store_cache()
        self.scan_cache.save_to_disk(robot_api=self)
        best_idx2, best_logit2 = self._select_best_from_cache(object_name, camera_id=camera_id)
        if best_idx2 < 0:
            print("[rescan] No valid view found after rescan.")
            return None
        entries2 = self.scan_cache.get()
        target_joints2 = entries2[best_idx2]['target_joints']
        print(f"[rescan-verify] Moving to pose index={best_idx2+1} to verify...")
        self._hold_and_wait_for_reach(target_joints2)
        time.sleep(0.3)
        verify_json2, verify_logit2 = self._verify_current_view(object_name, camera_id=camera_id)
        ratio2 = (verify_logit2 / best_logit2) if best_logit2 > 0 else 0.0
        print(f"[rescan-verify] Cached best logit={best_logit2:.3f}, live logit={verify_logit2:.3f}, ratio={ratio2:.3f}")
        if (
            verify_json2 is not None and
            len(verify_json2) >= 2 and 'box' in verify_json2[1] and
            verify_logit2 >= max(self.verify_abs_min_logit, self.verify_logit_drop_ratio * best_logit2)
        ):
            print("[rescan-verify] Target visible after rescan. Proceeding.")
            return verify_json2
        print("[rescan-verify] Target still not visible. Aborting.")
        return None

    def grasps_gen(self, object_name):
        self._log_timestamp("FUNCTION_CALL", f"grasps_gen(object_name='{object_name}')")
        data = self._segment_multi_view_for_grasp(object_name, num_views=3, camera_id=2)
        if len(data['pc']) == 0:
            alt = self._segment_for_grasp(object_name, camera_id=2)
            data = alt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # Save PCD data using dual system
        self._save_dual(data, "pcd", "json", timestamp, is_image=False)
        return sample_grasps(data['pc'], data['pc_color'])

    def _segment_multi_view_for_grasp(self, object_name, num_views=3, camera_id=2):
        mask_json = self._ensure_target_visible(object_name, camera_id=camera_id)
        if mask_json is None:
            return {'pc': [], 'pc_color': []}
        target_centroid = self._estimate_target_centroid_world(object_name, camera_id=camera_id)
        if target_centroid is None:
            return {'pc': [], 'pc_color': []}
        positions = self._plan_look_around_positions(target_centroid)
        fused_pts = []
        fused_cols = []
        _, viz_dir = self._get_trial_dirs()
        scan_dir = os.path.join(viz_dir, "look_around_images")
        try:
            os.makedirs(scan_dir, exist_ok=True)
            for name in os.listdir(scan_dir):
                try:
                    os.remove(os.path.join(scan_dir, name))
                except Exception:
                    pass
        except Exception:
            pass
        for k, p in enumerate(positions, start=1):
            if len(fused_pts) == num_views:
                break
            try:
                ok = self._move_camera_to(p, target_centroid)
                if not ok:
                    continue
                time.sleep(0.2)
                pts_w, cols = self._collect_points_at_current_view(object_name, camera_id=camera_id)
                if pts_w.shape[0] == 0:
                    continue
                fused_pts.append(pts_w)
                fused_cols.append(cols)
                try:
                    obs = self._get_processed_observation()
                    left_image = obs['observation']['camera']['image']['hand_camera'][0][..., :3][..., ::-1]
                    right_image = obs['observation']['camera']['image']['hand_camera'][1][..., :3][..., ::-1]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    # Save to BOTH latest_data and visualization using dual-save with subdirectory
                    self._save_dual(left_image, f"look_around_images/pose_{k}_left", "png", timestamp, context="look_around_scan")
                    self._save_dual(right_image, f"look_around_images/pose_{k}_right", "png", timestamp, context="look_around_scan")
                except Exception:
                    pass
            except Exception:
                continue
        if len(fused_pts) == 0:
            return {'pc': [], 'pc_color': []}
        pts_all = np.concatenate(fused_pts, axis=0)
        cols_all = np.concatenate(fused_cols, axis=0)
        if pts_all.shape[0] > 8000:
            idx = np.random.choice(pts_all.shape[0], 8000, replace=False)
            pts_all = pts_all[idx]
            cols_all = cols_all[idx]
        return {'pc': pts_all.tolist(), 'pc_color': cols_all.tolist()}

    def _segment_for_grasp(self, object_name, camera_id=2):
        mask_json = self._ensure_target_visible(object_name, camera_id=camera_id)
        if mask_json is None:
            return {'pc': [], 'pc_color': []}
        obs = self._get_processed_observation()
        color = obs['observation']['camera']['image']['hand_camera'][0][..., :3][..., ::-1]
        mask_json, masks = self._segment(color, object_name, camera_id)
        if len(mask_json) < 2:
            return {'pc': [], 'pc_color': []}
        mask = masks[0]
        xyz = self._get_pointcloud()
        points = xyz[mask == 1]
        data = self.env.get_observation()
        ext = np.array(data['camera_extrinsics'][f'{SN}_left'])
        camera_to_world_matrix = np.eye(4)
        camera_to_world_matrix[:3, :3] = euler_to_rmat(ext[3:6])
        camera_to_world_matrix[:3, 3] = ext[:3]
        homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = np.matmul(homogeneous_points, camera_to_world_matrix.T)[:, :3]
        color = color[mask == 1]
        index = np.random.choice(len(points), min(4096, len(points)), replace=False)
        data = {'pc': points[index].tolist(), 'pc_color': color[index].tolist()}
        return data

    def _get_pointcloud(self):
        """Wrist-camera point cloud (camera frame) of the current view."""
        return self._get_processed_observation()['observation']['camera']['pointcloud']['hand_camera'][0][..., :3]

    def _lift_pixel_to_world(self, pixel, mask=None):
        """World point for wrist-camera pixel (row, col), or None if there is no valid depth for it.

        Stereo depth has holes on shiny or textureless surfaces. Read the pixel from up to
        WRIST_DEPTH_FRAMES fresh frames; if it has no depth in any of them, place it on its
        camera ray at the median valid depth inside the object `mask`.
        """
        row, col = pixel
        for _ in range(WRIST_DEPTH_FRAMES):
            timestep = self._get_processed_observation()
            xyz_map = timestep['observation']['camera']['pointcloud']['hand_camera'][0][..., :3]
            xyz_camera = xyz_map[row, col]
            if np.isfinite(xyz_camera).all() and xyz_camera[2] > 0:
                break
        else:
            if mask is None:
                return None
            depth = xyz_map[mask == 1, 2]
            valid = depth[np.isfinite(depth) & (depth > 0)]
            if valid.size == 0 or valid.size < MIN_MASK_DEPTH_FRACTION * depth.size:
                return None
            z = float(np.median(valid))
            K = camera_matrix(timestep['intrinsics_dict']['hand_camera'][0])
            xyz_camera = np.array([(col - K[0, 2]) * z / K[0, 0], (row - K[1, 2]) * z / K[1, 1], z])
            print(f"[depth] No valid depth at pixel ({row}, {col}) in {WRIST_DEPTH_FRAMES} frames; "
                              f"using the median depth {z:.3f} m of {valid.size} object pixels.")
        ext = np.array(timestep['extrinsics_dict']['hand_camera'][0])
        return euler_to_rmat(ext[3:6]) @ xyz_camera + ext[:3]

    def record_frame(self):
        """Capture and write a frame to the video file.
        
        Records only from the configured external camera (varied_camera_2) to maintain a consistent view.
        """
        if not self._video_active or self.video_writer is None:
            return
            
        try:
            # Get frame from left camera only (single consistent view)
            frame_bgr = None

            # Read the configured external camera
            if frame_bgr is None:
                try:
                    camera_data = self.env.camera_reader.read_cameras()
                    if camera_data:
                        obs = camera_data[0] if isinstance(camera_data, (list, tuple)) else camera_data
                        img_dict = obs.get('image', {}) if isinstance(obs, dict) else {}
                        # Use the configured external camera (varied_camera_2) for consistent recording
                        if f'{VIDEO_CAMERA_ID}_left' in img_dict and img_dict[f'{VIDEO_CAMERA_ID}_left'] is not None:
                            frame_bgr = np.ascontiguousarray(img_dict[f'{VIDEO_CAMERA_ID}_left'][..., :3])
                except Exception:
                    # Main camera not available; proceed to fallback below
                    pass

            if frame_bgr is None:
                return
            
            # Ensure uint8 dtype for video encoding
            if frame_bgr.dtype != np.uint8:
                frame_bgr = frame_bgr.astype(np.uint8)
                
            # Resize only if dimensions were set and don't match
            if self._video_width and self._video_height:
                h, w = frame_bgr.shape[:2]
                if h != self._video_height or w != self._video_width:
                    frame_bgr = cv2.resize(frame_bgr, (self._video_width, self._video_height), interpolation=cv2.INTER_LINEAR)

            # Write frame
            with self._video_lock:
                if self.video_writer is not None:
                    self.video_writer.write(frame_bgr)
                    self.video_frame_count += 1
                    if self.video_frame_count == 1:
                        print(f"Recording started: {frame_bgr.shape}")
        except Exception as e:
            if self.video_frame_count == 0:
                print(f"Error recording frame: {e}")

    def start_video_recorder(self, fps=20, output_path=None):
        """Start video recording with direct-to-disk writing.
        
        Args:
            fps: Frames per second for the output video
            output_path: Optional path for the video file. If None, uses default path.
        """
        if self._video_active:
            print(f"Already recording to {self.video_path}")
            return  # Prevent starting multiple recordings
        
        # Set output path
        self.video_path = output_path if output_path else f"/tmp/robot_video_{int(time.time())}.mp4"
        print(f"Starting video recording to: {self.video_path}")
        
        # Get actual frame dimensions from camera
        test_frame = None
        try:
            camera_data = self.env.camera_reader.read_cameras()
            if camera_data:
                obs = camera_data[0] if isinstance(camera_data, (list, tuple)) else camera_data
                img_dict = obs.get('image', {}) if isinstance(obs, dict) else {}
                # Try to get the left camera frame to determine dimensions
                if f'{VIDEO_CAMERA_ID}_left' in img_dict and img_dict[f'{VIDEO_CAMERA_ID}_left'] is not None:
                    test_frame = img_dict[f'{VIDEO_CAMERA_ID}_left'][..., :3]
                elif img_dict:  # Fallback to any available camera
                    for img in img_dict.values():
                        if img is not None and len(img.shape) >= 3:
                            test_frame = img[..., :3]
                            break
        except Exception as e:
            print(f"Warning: Could not get test frame: {e}")
        
        # Set dimensions based on actual camera or use defaults
        if test_frame is not None:
            self._video_height, self._video_width = test_frame.shape[:2]
            print(f"Using camera dimensions: {self._video_width}x{self._video_height}")
        else:
            # Default dimensions if no frame available
            self._video_width, self._video_height = 640, 480
            print(f"Using default dimensions: {self._video_width}x{self._video_height}")
        
        # Initialize OpenCV writer with detected dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        with self._video_lock:
            self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, (self._video_width, self._video_height))
            if not self.video_writer.isOpened():
                print("Failed to create video writer")
                self.video_writer = None
                return
        
        self._video_active = True
        self.video_frame_count = 0
        
        # Start recording thread for continuous frame capture at specified FPS
        self._recorder_stop_evt.clear()
        interval = 1.0 / fps  # Time between frames
        
        def _recorder_loop():
            """Background thread that captures frames at regular intervals."""
            next_time = time.time()
            while not self._recorder_stop_evt.is_set():
                self.record_frame()
                # Calculate next frame time to maintain consistent FPS
                next_time += interval
                sleep_time = max(0, next_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        self._recorder_thread = threading.Thread(target=_recorder_loop, daemon=True)
        self._recorder_thread.start()
        print("Recording started")

    def stop_video_recorder(self):
        """Stop recording and save video file.
        
        This method is idempotent - safe to call multiple times.
        Called automatically on Ctrl+C via signal handler.
        """
        if not self._video_active:
            return  # Already stopped, nothing to do
            
        print("\nStopping video recorder...")
        
        # Stop the background recording thread
        if self._recorder_thread:
            self._recorder_stop_evt.set()
            self._recorder_thread.join(timeout=1.0)  # Wait for thread to finish
            self._recorder_thread = None
        
        # Release the video writer to finalize the file
        with self._video_lock:
            if self.video_writer:
                self.video_writer.release()  # Flush and close the video file
                self.video_writer = None
        
        # Check if video was saved
        if self.video_path and os.path.exists(self.video_path):
            file_size = os.path.getsize(self.video_path)
            print(f"Video saved: {self.video_path}")
            print(f"   Frames: {self.video_frame_count}, Size: {file_size/1024/1024:.2f} MB")
        else:
            print(f"Video file not found: {self.video_path}")
        
        # Reset state
        self._video_active = False
        self.video_path = None
        self.video_frame_count = 0
        self._video_width = None
        self._video_height = None

    def close_gripper(self):
        self._log_timestamp("FUNCTION_CALL", "close_gripper()")
        self.env._robot.update_gripper(1.0, velocity=False, blocking=True)
        time.sleep(1.0)
        requests.post(f"{service_url('curobo')}/toggle_collision_for_grasp", params={'flag': False}, timeout=10).raise_for_status()

    def _hold_and_wait_for_reach(self, target_joints):
        """Move to the target joints with blocking moves; True if within REACH_TOLERANCE.

        The robot server's joint controller has no integral term, so a steady load such as
        a sagging wrist leaves a steady error. A second move aimed past the target by the
        remaining error lands on it. Only blocking moves are sent: a non-blocking command
        runs later on the server and can cut the next blocking move short.
        """
        target = np.asarray(target_joints, dtype=float)
        command = target
        for _ in range(2):
            self.env._robot.update_joints(command, velocity=False, blocking=True)
            error = target - np.asarray(self.env._robot.get_joint_positions(), dtype=float)
            if np.linalg.norm(error) < REACH_TOLERANCE:
                return True
            correction = error if np.abs(error).max() <= MAX_REACH_CORRECTION else 0.0
            command = target + correction
        print(f"Target joints not reached; per-joint error (rad): {np.round(error, 3).tolist()}")
        return False

    def _sample_points_from_mask(self, text_prompt, camera_id=2):
        mask_json = self._ensure_target_visible(text_prompt, camera_id=camera_id)
        if mask_json is None:
            return []
        image = self._get_processed_observation()['observation']['camera']['image']['hand_camera'][0][..., :3][..., ::-1]
        mask_json, masks = self._segment(image, text_prompt, camera_id)
        if len(mask_json) < 2:
            return []
        mask = masks[0]
        points = self.generate_best_grid_from_mask(mask , (35, 45), 20)
        h, w = image.shape[:2]
        bbx = mask_json[1]['box']
        crop_img = image[int(bbx[1]-min(bbx[1], 10)):int(bbx[3]+min(h-bbx[3], 10)), int(bbx[0]-min(bbx[0], 10)):int(bbx[2]+min(w-bbx[2], 10))]
        scale = h // crop_img.shape[0]
        resized_points = (points - np.array([bbx[1]-min(bbx[1], 10), bbx[0]-min(bbx[0], 10)])) * scale
        crop_img = cv2.resize(crop_img, (crop_img.shape[1] * scale, crop_img.shape[0] * scale))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # Save crop image using dual system
        self._save_dual(crop_img, "crop_img", "png", timestamp)
        annotated = crop_img.copy()
        for i, (y, x) in enumerate(resized_points, 1):
            center = (int(x), int(y))
            cv2.circle(annotated, center, 17, (255, 0, 0), -1)
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(annotated, label, (center[0] - tw // 2, center[1] + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self._save_dual(annotated, "crop_img_annotated", "png", timestamp)
        self.logger.save_crop_image(Image.fromarray(annotated), text_prompt)
        return points

    def locate_object_part(self, object_name, part_description):
        self._log_timestamp("FUNCTION_CALL", f"locate_object_part(object_name='{object_name}', part_description='{part_description}')")
        """
        Given an object name and a part description, estimate the 3D position of that part of the object.
        Uses wrist-camera sampling and the part-selection VLM to pick point indices.
        Returns list of { '3d_position': [x,y,z], 'label': str, 'index': int }.
        """
        print(f"=== locate_object_part STARTED for object '{object_name}' with part description '{part_description}' ===")
        try:
            print("Calling _sample_points_from_mask...")
            pts = self._sample_points_from_mask(object_name)
            num_pts = 0 if pts is None else len(pts)
            print(f"_sample_points_from_mask returned: {num_pts} points")
        except Exception as e:
            print(f"ERROR in _sample_points_from_mask: {e}")
            import traceback
            traceback.print_exc()
            return []
        if pts is None or len(pts) == 0:
            print(f"No points found for object '{object_name}'. Segmentation may have failed.")
            return []
        position = self._get_pointcloud()
        data = self.env.get_observation()
        ext = data['camera_extrinsics'][f'{SN}_left']
        ext = np.array(ext)
        camera_to_world_matrix = np.eye(4)
        camera_to_world_matrix[:3, :3] = euler_to_rmat(ext[3:6])
        camera_to_world_matrix[:3, 3] = ext[:3]
        points = []
        for i, pt in enumerate(pts):
            xyz_camera = position[pt[0], pt[1]]
            if np.any(np.isnan(xyz_camera)) or np.any(np.isinf(xyz_camera)) or xyz_camera[2] <= 0:
                continue
            xyz_camera_homogeneous = np.array([xyz_camera[0], xyz_camera[1], xyz_camera[2], 1])
            xyz_world = np.matmul(xyz_camera_homogeneous, camera_to_world_matrix.T)
            if np.any(np.isnan(xyz_world[:3])) or np.any(np.isinf(xyz_world[:3])):
                continue
            points.append({'3d_position': xyz_world[:3], 'index': i+1})
        annotated = Image.open(self._load_latest("crop_img_annotated.png"))
        selected_points = vlm.select_points(annotated, points, part_description)
        index_to_point = {p['index']: p['3d_position'] for p in points}
        available_indices = sorted(index_to_point.keys())
        result = []
        for pt in selected_points:
            requested = pt.get('index')
            label = pt.get('label', '')
            if requested in index_to_point:
                result.append({'3d_position': index_to_point[requested], 'label': label, 'index': requested})
            else:
                if len(available_indices) == 0:
                    continue
                nearest = min(available_indices, key=lambda k: abs(k - int(requested)))
                print(f"Warning: selected index {requested} not available; using nearest available {nearest}.")
                result.append({'3d_position': index_to_point[nearest], 'label': label, 'index': nearest})
        return result

    def locate_objects(self, labels):
        """
        Find 3D positions of multiple objects using multi-view wrist image analysis.
        
        Strategy:
        1. Load cached wrist views (8 poses from different angles)
        2. Query the pointing VLM with ALL images at once for cross-view reasoning
        3. Move robot only to poses with detections and lift 2D points to 3D
        
        This is more efficient (1 API call vs N calls) and more accurate (model can 
        reason across all views to pick the best detection for each object).
        
        Args:
            labels: List of object names to locate (e.g., ["red apple", "blue mug"])
        
        Returns:
            List of dicts with found objects:
            [{'label': str, '2d_position': (y,x), '3d_position': [x, y, z]}, ...]
            
            If some objects are not found, returns partial results.
            The output will show which objects were found and which are missing.
        
        What to do if objects are missing:
            - You can switch to other perception method like locate_object_by_segmentation or locate_object_part
            - The scene may have changed - call robot.refresh_scan() to refresh the cache
            - Verify objects are actually in the scene
        
        Note: Does NOT automatically rescan. You decide when to rescan.
        """
        self._log_timestamp("FUNCTION_CALL", f"locate_objects(labels={labels})")
        
        if not labels:
            return []
        
        # Ensure scan cache exists
        if not self.scan_cache.valid:
            self.scan_cache.ensure_loaded_from_disk()
        if not self.scan_cache.valid:
            print(f"Creating wrist scan cache...")
            self._scan_workspace_store_cache()
            self.scan_cache.save_to_disk(robot_api=self)
        
        entries = self.scan_cache.get()
        if len(entries) == 0:
            print("No scan cache available")
            return []
        
        print(f"\n{'='*60}")
        print(f"Multi-View Search: Looking for {len(labels)} objects")
        print(f"  Objects: {labels}")
        print(f"  Available views: {len(entries)}")
        print(f"{'='*60}")
        
        # Query all views at once for better cross-view reasoning
        print(f"\nQuerying {vlm.POINTING_MODEL} with all {len(entries)} views...")
        
        wrist_images_rgb = [entry['left_image'] for entry in entries]
        
        pointing_results = vlm.point(wrist_images_rgb, labels)
        
        if len(pointing_results) == 0:
            print("No objects detected in any view")
            print(f"\n{'='*60}")
            print(f"Search Complete - No detections")
            print(f"  Requested: {labels}")
            print(f"  Found: []")
            print(f"  Missing: {labels}")
            print(f"{'='*60}")
            return []
        
        print(f"{vlm.POINTING_MODEL} detected {len(pointing_results)} objects across views")
        
        # Group detections by pose index
        detections_by_pose = {}
        for item in pointing_results:
            pose_idx = item.get('pose_index', 1)
            if pose_idx not in detections_by_pose:
                detections_by_pose[pose_idx] = []
            detections_by_pose[pose_idx].append(item)
        
        print(f"  Detections span {len(detections_by_pose)} different poses: {sorted(detections_by_pose.keys())}")
        
        # Process each pose with detections
        all_results = []
        all_overlays = []
        found_labels = set()
        
        for pose_idx in sorted(detections_by_pose.keys()):
            pose_detections = detections_by_pose[pose_idx]
            
            # Validate pose index
            if not (1 <= pose_idx <= len(entries)):
                print(f"\nInvalid pose_index {pose_idx}, skipping")
                continue
            
            entry = entries[pose_idx - 1]  # Convert to 0-based index
            
            print(f"\n>>> Processing pose {pose_idx} ({len(pose_detections)} detections)")
            for det in pose_detections:
                print(f"    - {det.get('label', 'unknown')}: {det.get('point', [])}")
            
            # Move robot to this pose
            target_joints = entry['target_joints']
            self._hold_and_wait_for_reach(target_joints)
            time.sleep(0.3)
            
            # Get live observation for 3D lifting
            obs = self._get_processed_observation()['observation']
            hand_images = obs['camera']['image']['hand_camera']
            live_left_rgb = hand_images[0][..., :3][..., ::-1]
            h, w = live_left_rgb.shape[:2]
            
            overlay = live_left_rgb.copy()

            view_had_detections = False
            
            # Lift 2D points to 3D
            for item in pose_detections:
                point = item.get('point', [])
                if len(point) != 2:
                    continue
                norm_y, norm_x = point
                label = item.get('label', '').strip()
                
                if not label:
                    continue
                
                # Match label to requested labels (fuzzy: allow Gemini to
                # drop articles like "the" or rephrase slightly)
                label_normalized = label.lower().strip()
                matching_label = None
                for orig_label in labels:
                    orig_norm = orig_label.lower().strip()
                    if orig_norm == label_normalized \
                       or label_normalized in orig_norm \
                       or orig_norm in label_normalized:
                        matching_label = orig_label
                        break
                
                if not matching_label:
                    print(f"    '{label}' doesn't match any requested label")
                    continue
                
                # Skip if already found
                if matching_label in found_labels:
                    print(f"    '{matching_label}' already found, skipping duplicate")
                    continue
                
                # Denormalize pixel coordinates
                py = int(round((float(norm_y) / 1000.0) * (h - 1)))
                px = int(round((float(norm_x) / 1000.0) * (w - 1)))
                
                if not (0 <= py < h and 0 <= px < w):
                    print(f"    '{label}' pixel out of bounds: ({py}, {px})")
                    continue
                
                # Lift to 3D
                xyz_world = self._lift_pixel_to_world((py, px))
                if xyz_world is None:
                    print(f"    '{label}' has invalid depth at ({py}, {px})")
                    continue

                # Success! Add to results
                all_results.append({
                    'label': matching_label,
                    '2d_position': (int(py), int(px)),
                    '3d_position': [float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])]
                })
                
                found_labels.add(matching_label)
                view_had_detections = True
                
                # Draw on overlay image
                cv2.circle(overlay, (px, py), radius=6, color=(255, 255, 255), thickness=-1)
                cv2.circle(overlay, (px, py), radius=5, color=(0, 0, 255), thickness=-1)
                
                label_x = px + 8
                label_y = py - 8
                (text_w, text_h), baseline = cv2.getTextSize(matching_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(overlay, 
                             (label_x - 2, label_y - text_h - 2),
                             (label_x + text_w + 2, label_y + baseline),
                             (59, 130, 246), -1)
                cv2.putText(overlay, matching_label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw on collage step image with label
                self.image_editor.draw_point_with_label(xyz_world.tolist(), matching_label)
                
                print(f"    {matching_label}: pixel ({py}, {px}) -> world {xyz_world.tolist()}")
            
            if view_had_detections:
                all_overlays.append((pose_idx, overlay))
        
        # Save all annotated wrist images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        for pose_idx, overlay in all_overlays:
            self._save_dual(overlay, f"wrist_points_view{pose_idx}", "png", timestamp, context="wrist_points_annotation")
        
        # Save mapping JSON
        if len(all_results) > 0:
            mapping = {
                "labels": labels,
                "found_labels": [r['label'] for r in all_results],
                "world_points": {r['label']: r['3d_position'] for r in all_results}
            }
            self._save_dual(mapping, "wrist_points_world", "json", timestamp, is_image=False, context="wrist_points_projection")
        
        # Summary
        missing_labels = [label for label in labels if label not in found_labels]
        
        print(f"\n{'='*60}")
        print(f"Multi-View Search Complete")
        print(f"  Requested: {labels}")
        print(f"  Found: {list(found_labels)}")
        print(f"  Missing: {missing_labels}")
        print(f"  Success rate: {len(found_labels)}/{len(labels)}")
        print(f"{'='*60}")
        
        if len(missing_labels) > 0:
            print(f"\nCould not find {len(missing_labels)} objects: {missing_labels}")
            print(f"   Suggestions:")
            print(f"   - The scene may have changed. Call robot.refresh_scan() to refresh cache")
            print(f"   - Try other perception methods (locate_object_by_segmentation, locate_object_part)")
            print(f"   - Verify objects are actually in the scene")
        
        return all_results

    def get_robot_state(self):
        self._log_timestamp("FUNCTION_CALL", "get_robot_state()")
        state_dict = dict()
        robot_state = self.env._robot.get_robot_state()
        pos = robot_state[0]['cartesian_position'][:3]
        ori = robot_state[0]['cartesian_position'][3:]
        gripper_fraction_closed = None
        try:
            gripper_fraction_closed = float(self.env._robot.get_gripper_position())
            print("gripper_fraction_closed", gripper_fraction_closed)
        except Exception:
            pass
        rotate_mat = euler_to_rmat(ori)
        gripper_center_pos = pos + rotate_mat @ GRIPPER_FLANGE_OFFSET
        state_dict["robot_state"] = {
            "position": gripper_center_pos,
            "orientation": ori,
            "gripper_fraction_closed": gripper_fraction_closed,
        }
        return state_dict

    def get_image(self):
        self._log_timestamp("FUNCTION_CALL", "get_image()")
        obs = self._get_processed_observation()['observation']
        images = []
        for key in obs['camera']['image'].keys():
            images.append(obs['camera']['image'][key][0][..., :3][..., ::-1])
        if len(images) == 0:
            # Return a blank image if no cameras are available
            return np.zeros((480, 640, 3), dtype=np.uint8)
        images = np.concatenate(images, axis=1)
        return images

    def _current_wrist_view(self):
        """Stereo depth view of the current wrist image, saved with the trial."""
        cam = str(hand_camera_id)
        images = self.env.camera_reader.read_cameras()[0]['image']
        left = cv2.cvtColor(images[f'{cam}_left'], cv2.COLOR_BGRA2RGB)
        right = cv2.cvtColor(images[f'{cam}_right'], cv2.COLOR_BGRA2RGB)
        calibration = self.env.camera_reader.get_camera(cam).get_calibration()
        depth = stereo_depth(left, right, calibration)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._save_dual(left, "rgb_1", "png", timestamp, context="collision_view")
        self._save_dual(right, "right_rgb_1", "png", timestamp, context="collision_view")
        self._save_dual(depth, "depth_1", "npy", timestamp, is_image=False, context="collision_view")
        ext = np.array(self.env.get_observation()['camera_extrinsics'][f'{cam}_left'])
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = euler_to_rmat(ext[3:6])
        cam_pose[:3, 3] = ext[:3]
        return {'depth': depth, 'intrinsics': stereo_geometry(calibration)[0], 'cam_pose': cam_pose}

    def _update_collision_world(self):
        """Fuse the wrist scan's depth views into cuRobo's world, or the current view without a scan."""
        views = [entry for entry in self.scan_cache.get()
                 if all(entry.get(key) is not None for key in ('depth', 'intrinsics', 'cam_pose'))]
        fuse_observation(views or [self._current_wrist_view()])

    def move_to_pose(self, position, orientation):
        self._log_timestamp("FUNCTION_CALL", f"move_to_pose(position={position}, orientation={orientation})")
        if self.use_motionplanner:
            self._update_collision_world()
            joint_positions = self.env._robot.get_joint_positions()
            rotation_mat = euler_to_rmat(orientation)
            flange_pos = position - rotation_mat @ GRIPPER_FLANGE_OFFSET
            goal_pose = np.zeros((4, 4))
            goal_pose[:3, 3] = np.array(flange_pos)
            goal_pose[:3, :3] = rotation_mat
            goal_pose[3, 3] = 1.0
            data = {
                "goal_pose": goal_pose.tolist(),
                "start_state": joint_positions.tolist()
            }
            response = requests.post(f"{service_url('curobo')}/plan", json=data, timeout=(10, 120))
            response.raise_for_status()
            if not response.json()['success']:
                print(f"Motion planning failed, cannot move gripper to the desired pose {position}, {orientation}.")
                return False
            dt = response.json()['dt']
            poses = response.json()['trajectory']
            skip_factor = 3
            print(f"Executing trajectory with {len(poses)} total points")
            print(f"Using every {skip_factor}th waypoint = {len(poses)//skip_factor} network commands")
            for i in range(0, len(poses), skip_factor):
                pose = poses[i]
                self.env._robot.update_joints(np.array(pose), velocity=False, blocking=False)
                time.sleep(dt * skip_factor)
            return True
        else:
            position = np.array(position)
            orientation = np.array(orientation)
            rotation_mat = euler_to_rmat(orientation)
            flange_pos = position - rotation_mat @ GRIPPER_FLANGE_OFFSET
            pose = np.concatenate((flange_pos, orientation))
            for j in range(100):
                curr_pose = self.env._robot.get_ee_pose()
                cartesian_delta = pose_diff(pose, curr_pose)
                if np.linalg.norm(cartesian_delta) < 0.02:
                    return True
                self.env._robot.update_pose(pose, velocity=False, blocking=False)
                time.sleep(0.1)
            print(f"Could not reach the desired pose {position}, {orientation}.")
            return False

    def propose_grasps(self, object_name):
        self._log_timestamp("FUNCTION_CALL", f"propose_grasps(object_name='{object_name}')")
        grasps = self.grasps_gen(object_name)
        all_pos = grasps[:, :3, 3]
        all_rot_mat = grasps[:, :3, :3]
        result = []
        for i in range(all_pos.shape[0]):
            pos = all_pos[i]
            rot_mat = all_rot_mat[i]
            euler = rmat_to_euler(rot_mat)
            R = np.array(
                [[0., -1., 0.],
                [1., 0., 0.],
                [0., 0., 1.]]
            )
            rot_mat = euler_to_rmat(euler)
            rot_mat = rot_mat @ R
            euler = rmat_to_euler(rot_mat)
            GRIPPER_FLANGE_OFFSET_2 = np.array([-0.00, -0.00, 0.10])
            pos = pos + rot_mat @ GRIPPER_FLANGE_OFFSET_2
            result.append((pos, euler))
        return result

    def measure_distance(self, position1, position2):
        self._log_timestamp("FUNCTION_CALL", f"measure_distance(position1={position1}, position2={position2})")
        position1 = np.array(position1)
        position2 = np.array(position2)
        return np.linalg.norm(position1 - position2)

    def measure_vector(self, position1, position2):
        self._log_timestamp("FUNCTION_CALL", f"measure_vector(position1={position1}, position2={position2})")
        position1 = np.array(position1)
        position2 = np.array(position2)
        return position2 - position1

    def euler_to_rotation_matrix(self, euler):
        self._log_timestamp("FUNCTION_CALL", f"euler_to_rotation_matrix(euler={euler})")
        euler = np.array(euler)
        return euler_to_rmat(euler)

    def rotation_matrix_to_euler(self, rotation_matrix):
        self._log_timestamp("FUNCTION_CALL", f"rotation_matrix_to_euler(rotation_matrix={rotation_matrix.tolist() if isinstance(rotation_matrix, np.ndarray) else rotation_matrix})")
        return rmat_to_euler(np.array(rotation_matrix))

    def measure_relative_rotation(self, vector1, vector2):
        self._log_timestamp("FUNCTION_CALL", f"measure_relative_rotation(vector1={vector1}, vector2={vector2})")
        vector1 = np.array(vector1) / np.linalg.norm(vector1)
        vector2 = np.array(vector2) / np.linalg.norm(vector2)
        v = np.cross(vector1, vector2)
        c = np.dot(vector1, vector2)
        if c > 0.9999:
            return np.array([0., 0., 0.])
        if c < -0.9999:
            axis = np.array([1., 0., 0.])
            if abs(vector1[0]) > 0.9:
                axis = np.array([0., 1., 0.])
            axis = axis - vector1 * np.dot(axis, vector1)
            axis = axis / np.linalg.norm(axis)
            return axis * np.pi
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))
        euler = rmat_to_euler(R)
        return euler

    def rotate_by(self, current_orientation, rotation):
        self._log_timestamp("FUNCTION_CALL", f"rotate_by(current_orientation={current_orientation}, rotation={rotation})")
        current_orientation = np.array(current_orientation)
        rotation = np.array(rotation)
        rmat1 = euler_to_rmat(current_orientation)
        rmat2 = euler_to_rmat(rotation)
        rmat = rmat2 @ rmat1
        return rmat_to_euler(rmat)

    def execute_grasp(self, grasps):
        self._log_timestamp("FUNCTION_CALL", f"execute_grasp(grasps={grasps})")
        joint_positions = self.env._robot.get_joint_positions()
        goal_pose = np.zeros((len(grasps), 4, 4))
        for i, grasp in enumerate(grasps):
            grasp_position, grasp_orientation = grasp
            goal_pose[i, :3, 3] = grasp_position - euler_to_rmat(grasp_orientation) @ GRIPPER_FLANGE_OFFSET
            goal_pose[i, :3, :3] = euler_to_rmat(grasp_orientation)
            goal_pose[i, 3, 3] = 1.0
        if self.use_motionplanner:
            data = {
                "goal_pose": goal_pose.tolist(),
                "start_state": joint_positions.tolist()
            }
            self._update_collision_world()
            response = requests.post(f"{service_url('curobo')}/plan_grasp", json=data, timeout=(10, 120))
            response.raise_for_status()
            dt = response.json()['dt'] if 'dt' in response.json() else 0.1
            poses = response.json().get('trajectory', [])
            if not response.json().get('success', False) or not poses:
                print("Motion planning for grasp failed")
                return
            skip_factor = 3
            print(f"Executing trajectory with {len(poses)} total points")
            print(f"Using every {skip_factor}th waypoint = {len(poses)//skip_factor} network commands")
            for i in range(0, len(poses), skip_factor):
                pose = poses[i]
                self.env._robot.update_joints(np.array(pose), velocity=False, blocking=False)
                time.sleep(dt * skip_factor)
            time.sleep(1.0)
            self.close_gripper()
            time.sleep(1.0)
        else:
            grasp_position, grasp_orientation = grasps[0]
            self.move_to_pose(grasp_position, grasp_orientation)
            time.sleep(1.0)
            self.close_gripper()

    def open_gripper(self):
        self._log_timestamp("FUNCTION_CALL", "open_gripper()")
        self.env._robot.update_gripper(0.0, velocity=False, blocking=True)
        requests.post(f"{service_url('curobo')}/toggle_collision_for_grasp", params={'flag': True}, timeout=10).raise_for_status()

    def reset(self):
        self._log_timestamp("FUNCTION_CALL", "reset()")
        self.env.reset()

    def scan_workspace(self):
        """
        Perform initial wrist scan to cache views of the scene.
        
        This creates a cache of 8 wrist camera views from different angles.
        Use this at the beginning of a task to prepare for object detection.
        
        For rescanning (invalidating old cache), use refresh_scan() instead.
        """
        self._log_timestamp("FUNCTION_CALL", "scan_workspace()")
        self.is_scanning_wrist = True
        try:
            self._scan_workspace_store_cache()
            self.scan_cache.save_to_disk(robot_api=self)
            print("Wrist scan completed and saved")
        finally:
            self.is_scanning_wrist = False
    
    def refresh_scan(self):
        """
        Invalidate the existing scan cache and perform a fresh wrist scan.
        
        Use this when:
        - Objects have moved since the last scan
        - locate_objects failed to find some objects
        - The scene has changed significantly
        - Lighting conditions changed
        
        This will discard the old cache and create a new one from scratch.
        After rescanning, you can call locate_objects again to find objects.
        """
        # clean the curobo map
        requests.post(f"{service_url('curobo')}/clean_map")
        self._log_timestamp("FUNCTION_CALL", "refresh_scan()")
        print("\nRescanning wrist views (invalidating old cache)...")
        self.is_scanning_wrist = True
        try:
            self.scan_cache.invalidate()
            self._scan_workspace_store_cache()
            self.scan_cache.save_to_disk(robot_api=self)
            print(f"Rescan completed - new cache saved with {len(self.scan_cache.get())} views")
        finally:
            self.is_scanning_wrist = False


class ImageEditApi:
    """Interface for editing the image for a better understanding."""
    def __init__(self, env):
        self.env = env
        self.ouput_images = None
        self.timeprocesser = TimestepProcesser(ignore_action=True)
        self.edit_action_buffer = []

    def _get_processed_observation(self):
        obs = self.env.get_observation()
        timestep = {'observation': obs}
        timestep = self.timeprocesser.forward(timestep)
        obs = timestep
        return obs

    def get_raw_images(self):
        """Build a two-view horizontal collage (wrist | side) and store per-view metadata."""
        timestep = self._get_processed_observation()
        obs = timestep['observation']
        intrinsics_dict = timestep.get('intrinsics_dict', {})
        extrinsics_dict = timestep.get('extrinsics_dict', {})
        
        views_meta = []
        collage_parts = []
        x_offset = 0
        
        # View 1: wrist left (hand_camera[0])
        if 'hand_camera' in obs['camera']['image'] and len(obs['camera']['image']['hand_camera']) > 0:
            wrist_left = obs['camera']['image']['hand_camera'][0][..., :3][..., ::-1]
            if 'hand_camera' in intrinsics_dict and 'hand_camera' in extrinsics_dict:
                intr_h = intrinsics_dict['hand_camera'][0]
                extr_h = extrinsics_dict['hand_camera'][0]
                h, w = wrist_left.shape[:2]
                views_meta.append({
                    'name': 'hand_camera_left',
                    'intrinsic': np.array(intr_h, dtype=np.float32),
                    'extrinsic': np.array(extr_h, dtype=np.float32),
                    'x_offset': int(x_offset),
                    'width': int(w),
                    'height': int(h),
                })
                collage_parts.append(wrist_left)
                x_offset += w
        
        # View 2: side camera left (varied_camera[0])
        if 'varied_camera' in obs['camera']['image'] and len(obs['camera']['image']['varied_camera']) > 0:
            side_left = obs['camera']['image']['varied_camera'][0][..., :3][..., ::-1]
            if 'varied_camera' in intrinsics_dict and 'varied_camera' in extrinsics_dict:
                intr_v = intrinsics_dict['varied_camera'][0]
                extr_v = extrinsics_dict['varied_camera'][0]
                h, w = side_left.shape[:2]
                views_meta.append({
                    'name': 'varied_camera0_left',
                    'intrinsic': np.array(intr_v, dtype=np.float32),
                    'extrinsic': np.array(extr_v, dtype=np.float32),
                    'x_offset': int(x_offset),
                    'width': int(w),
                    'height': int(h),
                })
                collage_parts.append(side_left)
                x_offset += w
        
        # Fallback if no views available
        if len(collage_parts) == 0:
            self.views_meta = []
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Pad to uniform height
        target_h = max(part.shape[0] for part in collage_parts)
        padded_parts = []
        for part in collage_parts:
            h, w = part.shape[:2]
            if h < target_h:
                pad = target_h - h
                part = np.pad(part, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_parts.append(part)
        
        # Update metadata with final dimensions
        for view in views_meta:
            view['height'] = int(target_h)
        
        # Create collage and store metadata
        collage = np.concatenate(padded_parts, axis=1)
        self.views_meta = views_meta
        
        return collage

    def world_to_image(self, position, intrinsic, extrinsic):
        position = np.array(position, dtype=np.float32)
        position_homogeneous = np.append(position, 1.0)
        camera_position = np.linalg.inv(extrinsic) @ position_homogeneous
        if camera_position[2] <= 0:
            return None, None
        image_position = intrinsic @ camera_position[:3]
        image_position /= image_position[2]
        z_cam = camera_position[2]
        fx = intrinsic[0, 0]
        scale = fx / z_cam
        return image_position[:2], scale

    def draw_point(self, position):
        if np.any(np.isnan(position)) or np.any(np.isinf(position)):
            print(f"Warning: Skipping drawing point with invalid coordinates: {position}")
            return
        self.edit_action_buffer.append((self._draw_point, position))

    def draw_point_with_label(self, position, label):
        """Draw a 3D point with a text label on the collage."""
        if np.any(np.isnan(position)) or np.any(np.isinf(position)):
            print(f"Warning: Skipping drawing labeled point with invalid coordinates: {position}")
            return
        self.edit_action_buffer.append((self._draw_point_with_label, position, label))

    def draw_pose(self, position, orientation):
        self.edit_action_buffer.append((self._draw_pose, position, orientation))

    def _draw_point(self, position):
        """Draw a 3D world point on the collage by projecting it into each view."""
        image = self.ouput_images.copy()
        views = getattr(self, 'views_meta', [])
        
        if not views:
            # Fallback: single-view mode using varied_camera[0]
            timestep = self._get_processed_observation()
            intrinsic = timestep["intrinsics_dict"]['varied_camera'][0]
            extrinsic = timestep['extrinsics_dict']['varied_camera'][0]
            extrinsic_mat = np.eye(4, dtype=np.float32)
            extrinsic_mat[:3, :3] = euler_to_rmat(extrinsic[3:6])
            extrinsic_mat[:3, 3] = extrinsic[:3]
            
            xy, scale = self.world_to_image(position, intrinsic, extrinsic_mat)
            if xy is not None and not np.any(np.isnan(xy)):
                x, y = int(round(xy[0])), int(round(xy[1]))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    radius = max(3, int(0.01 * scale + 1))
                    cv2.circle(image, (x, y), radius, (0, 255, 0), thickness=-1)
            self.ouput_images = image
            return
        
        # Multi-view: project into each view and draw with correct offset
        for view in views:
            intrinsic = view['intrinsic']
            extrinsic_6d = view['extrinsic']
            x_offset = view['x_offset']
            
            # Build camera-to-world matrix from 6D extrinsic [pos, euler]
            extrinsic_mat = np.eye(4, dtype=np.float32)
            extrinsic_mat[:3, :3] = euler_to_rmat(extrinsic_6d[3:6])
            extrinsic_mat[:3, 3] = extrinsic_6d[:3]
            
            # Project world point to this view's pixel coordinates
            xy, scale = self.world_to_image(position, intrinsic, extrinsic_mat)
            if xy is None or np.any(np.isnan(xy)):
                continue
            
            # Add x_offset to map from view coordinates to collage coordinates
            x_collage = int(round(xy[0])) + x_offset
            y_collage = int(round(xy[1]))
            
            if 0 <= x_collage < image.shape[1] and 0 <= y_collage < image.shape[0]:
                radius = max(3, int(0.01 * scale + 1))
                cv2.circle(image, (x_collage, y_collage), radius, (0, 255, 0), thickness=-1)
        
        self.ouput_images = image

    def _draw_point_with_label(self, position, label):
        """Draw a 3D world point with a text label on the collage by projecting it into each view."""
        image = self.ouput_images.copy()
        views = getattr(self, 'views_meta', [])
        
        if not views:
            # Fallback: single-view mode using varied_camera[0]
            timestep = self._get_processed_observation()
            intrinsic = timestep["intrinsics_dict"]['varied_camera'][0]
            extrinsic = timestep['extrinsics_dict']['varied_camera'][0]
            extrinsic_mat = np.eye(4, dtype=np.float32)
            extrinsic_mat[:3, :3] = euler_to_rmat(extrinsic[3:6])
            extrinsic_mat[:3, 3] = extrinsic[:3]
            
            xy, scale = self.world_to_image(position, intrinsic, extrinsic_mat)
            if xy is not None and not np.any(np.isnan(xy)):
                x, y = int(round(xy[0])), int(round(xy[1]))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    radius = max(3, int(0.01 * scale + 1))
                    # Draw white border + red center
                    cv2.circle(image, (x, y), radius + 1, (255, 255, 255), thickness=-1)
                    cv2.circle(image, (x, y), radius, (0, 0, 255), thickness=-1)
                    
                    # Add label with blue background
                    label_x = x + 8
                    label_y = y - 8
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(image, 
                                 (label_x - 2, label_y - text_h - 2),
                                 (label_x + text_w + 2, label_y + baseline),
                                 (255, 130, 59),
                                 -1)
                    cv2.putText(image, label, (label_x, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            self.ouput_images = image
            return
        
        # Multi-view: project into each view and draw with correct offset
        for view in views:
            intrinsic = view['intrinsic']
            extrinsic_6d = view['extrinsic']
            x_offset = view['x_offset']
            
            # Build camera-to-world matrix from 6D extrinsic [pos, euler]
            extrinsic_mat = np.eye(4, dtype=np.float32)
            extrinsic_mat[:3, :3] = euler_to_rmat(extrinsic_6d[3:6])
            extrinsic_mat[:3, 3] = extrinsic_6d[:3]
            
            # Project world point to this view's pixel coordinates
            xy, scale = self.world_to_image(position, intrinsic, extrinsic_mat)
            if xy is None or np.any(np.isnan(xy)):
                continue
            
            # Add x_offset to map from view coordinates to collage coordinates
            x_collage = int(round(xy[0])) + x_offset
            y_collage = int(round(xy[1]))
            
            if 0 <= x_collage < image.shape[1] and 0 <= y_collage < image.shape[0]:
                radius = max(3, int(0.01 * scale + 1))
                # Draw white border + red center
                cv2.circle(image, (x_collage, y_collage), radius + 1, (255, 255, 255), thickness=-1)
                cv2.circle(image, (x_collage, y_collage), radius, (0, 0, 255), thickness=-1)
                
                # Add label with blue background
                label_x = x_collage + 8
                label_y = y_collage - 8
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                # Make sure label stays within image bounds
                if label_x + text_w + 2 > image.shape[1]:
                    label_x = x_collage - text_w - 10
                if label_y - text_h - 2 < 0:
                    label_y = y_collage + text_h + 10
                    
                cv2.rectangle(image, 
                             (label_x - 2, label_y - text_h - 2),
                             (label_x + text_w + 2, label_y + baseline),
                             (255, 130, 59),  # Blue in BGR
                             -1)
                cv2.putText(image, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        self.ouput_images = image

    def _draw_pose(self, position, orientation):
        """Draw a 3D pose (axes) on the collage by projecting into each view."""
        position = np.array(position, dtype=np.float32)
        orientation = np.array(orientation, dtype=np.float32)
        
        image = self.ouput_images.copy()
        views = getattr(self, 'views_meta', [])
        distCoeffs = np.zeros(5, dtype=np.float32)
        
        if not views:
            # Fallback: single-view mode using varied_camera[0]
            timestep = self._get_processed_observation()
            intrinsic = timestep["intrinsics_dict"]['varied_camera'][0]
            extrinsic = timestep['extrinsics_dict']['varied_camera'][0]
            extrinsic_mat = np.eye(4, dtype=np.float32)
            extrinsic_mat[:3, :3] = euler_to_rmat(extrinsic[3:6])
            extrinsic_mat[:3, 3] = extrinsic[:3]
            
            # Transform pose to camera frame
            world_to_cam = np.linalg.inv(extrinsic_mat)
            rotate = euler_to_rmat(orientation)
            pose_mat = np.eye(4, dtype=np.float32)
            pose_mat[:3, :3] = rotate
            pose_mat[:3, 3] = position
            cam_pose = world_to_cam @ pose_mat
            
            rvec, _ = cv2.Rodrigues(cam_pose[:3, :3])
            tvec = cam_pose[:3, 3].reshape(3, 1)
            cv2.drawFrameAxes(image, intrinsic, distCoeffs, rvec, tvec, 0.10, 2)
            self.ouput_images = image
            return
        
        # Multi-view: draw axes on each view's region
        for view in views:
            intrinsic = view['intrinsic']
            extrinsic_6d = view['extrinsic']
            x_offset = view['x_offset']
            
            # Build camera-to-world matrix
            extrinsic_mat = np.eye(4, dtype=np.float32)
            extrinsic_mat[:3, :3] = euler_to_rmat(extrinsic_6d[3:6])
            extrinsic_mat[:3, 3] = extrinsic_6d[:3]
            
            # Transform pose to camera frame
            world_to_cam = np.linalg.inv(extrinsic_mat)
            rotate = euler_to_rmat(orientation)
            pose_mat = np.eye(4, dtype=np.float32)
            pose_mat[:3, :3] = rotate
            pose_mat[:3, 3] = position
            cam_pose = world_to_cam @ pose_mat
            
            rvec, _ = cv2.Rodrigues(cam_pose[:3, :3])
            tvec = cam_pose[:3, 3].reshape(3, 1)
            
            # Adjust intrinsic principal point for the view's position in collage
            K_view = intrinsic.copy()
            K_view[0, 2] += float(x_offset)
            
            cv2.drawFrameAxes(image, K_view, distCoeffs, rvec, tvec, 0.10, 2)
        
        self.ouput_images = image

    def feedback(self):
        self.ouput_images = self.get_raw_images()
        for action in self.edit_action_buffer:
            if len(action) == 2:
                action[0](action[1])
            elif len(action) == 3:
                action[0](action[1], action[2])
        self.edit_action_buffer = []
        return self.ouput_images
