# cuRobo motion-planning service

Collision-free motion and grasp planning with [cuRobo](https://github.com/NVlabs/curobo) and an
nvblox collision world fused from depth views. FastAPI on `:8004`.

- `POST /plan` `{goal_pose, start_state, constrained}` → `{success, trajectory, dt}`
- `POST /plan_grasp` `{goal_pose, start_state}` → `{success, trajectory, dt}`
- `POST /fuse_observation`: file `views`, an `.npz` of `depths` (N×H×W meters), `intrinsics`
  (N×3×3) and camera-to-world `camera_poses` (N×4×4) → `{success, views}`
- `POST /clean_map`, `POST /toggle_collision_for_grasp?flag=<bool>` → `{success}`

`curobo_overlay/` holds Maestro's only additions to cuRobo, copied onto its source tree by the
Dockerfile: `planner.py` (the `CuroboPlanner` behind the endpoints), the Franka + Robotiq 2F-85
robot model (`franka_robotiq.yml`, collision spheres, URDF and meshes). The image builds
nvblox and nvblox_torch and pins every upstream commit and the packages in `requirements.txt`.
`CUROBO_ROBOT_TYPE` selects `franka_robotiq` (default; the gripper the client is set up for) or
`franka` (Franka Hand, which also needs the client changes in
[HARDWARE.md](../../docs/HARDWARE.md)). Run it with `docker compose up -d --build curobo` from the repository root; see
[SETUP_SERVICES.md](../../docs/SETUP_SERVICES.md).
