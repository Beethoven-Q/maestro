# GraspGen grasp service

6-DoF grasp candidates from an object point cloud with [GraspGen](https://github.com/NVlabs/GraspGen)
(Franka Panda gripper checkpoint). FastAPI on `:8006`, meshcat viewer on `:7000`.

- `POST /grasps`: file `cloud`, an `.npz` of world-frame `points` (N×3) and `colors` (N×3 uint8)
  → `.npz` of `grasps` (M×4×4 world-frame gripper poses) and `scores` (M).
- `GET /health` → ready once the model has loaded.
- `http://<host>:7000/static/` shows the latest request's point cloud and grasps.

The image pins GraspGen's commit (plus `python39.patch` and `meshcat-transform.patch`), the
packages in `requirements.txt`, and the checkpoint files by SHA-256.
Run it with `docker compose up -d --build graspgen` from the repository root; see
[SETUP_SERVICES.md](../../docs/SETUP_SERVICES.md).
