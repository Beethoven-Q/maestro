import os
from cv2 import aruco

# Robot addresses and camera serials come from config/maestro.yaml.
try:
    from maestro.config import cfg
    _robot = dict(cfg.robot) if cfg.robot else {}
    _cams = dict(cfg.cameras) if cfg.cameras else {}
except Exception:
    _robot, _cams = {}, {}


def _get(d, key, default):
    return d[key] if key in d else default


# Robot Params #
nuc_ip = _get(_robot, "nuc_ip", "172.16.0.4")
robot_ip = _get(_robot, "robot_ip", "172.16.0.2")
laptop_ip = _get(_robot, "laptop_ip", "172.16.0.1")
sudo_password = os.environ.get("ROBOT_SUDO_PASSWORD", "")
robot_type = _get(_robot, "model", "fr3")              # 'panda' or 'fr3'
robot_serial_number = _get(_robot, "serial_number", "")

# Camera IDs (ZED serials), from config/maestro.yaml -> cameras #
def _camera_id(key):
    value = _cams.get(key)
    return None if value is None else str(value)


hand_camera_id = _camera_id("hand_camera_id")
varied_camera_1_id = _camera_id("varied_camera_1_id")
varied_camera_2_id = _camera_id("varied_camera_2_id")


# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 12
CHARUCOBOARD_CHECKER_SIZE = 0.030
CHARUCOBOARD_MARKER_SIZE = 0.023
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)
# ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100) # Change to this if you are using roboarena_evaluator to test the policy

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = ""

# Code Version [DONT CHANGE] #
droid_version = "1.3"
