from threading import RLock

from droid.robot_env import RobotEnv as BaseRobotEnv


class RobotEnvThreadSafe(BaseRobotEnv):
    """Thread-safe variant of RobotEnv for concurrent VLM + data logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_lock = RLock()

    def get_state(self):
        with self._obs_lock:
            return super().get_state()

    def read_cameras(self):
        with self._obs_lock:
            return super().read_cameras()

    def get_observation(self):
        with self._obs_lock:
            return super().get_observation()

