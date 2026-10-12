"""Maestro GraspGen grasp-sampling server (FastAPI).

POST /grasps with an .npz file `cloud` holding an object's world-frame `points` (N, 3) and
their RGB `colors` (N, 3 uint8). Responds with an .npz of `grasps` (M, 4, 4 gripper poses in
the world frame) and their `scores` (M,). Each request's cloud and grasps are
drawn in the built-in meshcat viewer at http://<host>:7000/static/.

Configured via environment variables:
  GRASPGEN_PORT     listen port (default 8006)
  GRASPGEN_CONFIG   GraspGen checkpoint config, e.g. graspgen_franka_panda.yml (required;
                    the .pth files it names sit next to it)
"""
import io
import os

import meshcat
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import visualize_grasp, visualize_pointcloud

PORT = int(os.environ.get("GRASPGEN_PORT", "8006"))
CONFIG = os.environ.get("GRASPGEN_CONFIG")
if not CONFIG:
    raise EnvironmentError("GRASPGEN_CONFIG must point to the GraspGen checkpoint config (.yml)")

grasp_cfg = load_grasp_cfg(CONFIG)
GRIPPER = grasp_cfg.data.gripper_name
sampler = GraspGenSampler(grasp_cfg)
viewer = meshcat.Visualizer()  # starts its own meshcat server; web page on port 7000
print(f"[GraspGen] Model loaded from {CONFIG}; viewer at {viewer.url()}")

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ready", "service": "graspgen"}


@app.post("/grasps")
def grasps(cloud: UploadFile):
    with np.load(io.BytesIO(cloud.file.read()), allow_pickle=False) as data:
        points, colors = data["points"], data["colors"]
    poses, scores = GraspGenSampler.run_inference(
        points, sampler, grasp_threshold=0.8, num_grasps=1500, topk_num_grasps=1000
    )
    poses = poses.cpu().numpy() if len(poses) else np.zeros((0, 4, 4), dtype=np.float32)
    scores = scores.cpu().numpy() if len(scores) else np.zeros(0, dtype=np.float32)
    if len(scores):
        print(f"[GraspGen] {len(scores)} grasps, scores {scores.min():.3f}-{scores.max():.3f}")

    viewer.delete()
    visualize_pointcloud(viewer, "pc_obj", points, colors, size=0.005)
    for j, pose in enumerate(poses):
        visualize_grasp(viewer, f"GraspGen/{j:03d}/grasp", pose, color=[0, 185, 0],
                        gripper_name=GRIPPER, linewidth=1.5)

    buffer = io.BytesIO()
    np.savez(buffer, grasps=poses, scores=scores)
    return Response(buffer.getvalue(), media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
