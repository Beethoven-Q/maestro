"""Maestro cuRobo motion-planning server (FastAPI).

Collision-free motion planning + grasp planning + nvblox world fusion over HTTP.
Everything is configured via environment variables (no hardcoded paths):

  CUROBO_PORT        listen port                       (default 8004)
  CUROBO_ROBOT_TYPE  curobo robot config               (default 'franka_robotiq', the Robotiq
                                                         2F-85 the client is set up for)

Requires cuRobo with this directory's curobo_overlay/ (the CuroboPlanner and the
franka_robotiq robot model) and nvblox; see the Dockerfile.
"""
import io
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from curobo.planner import CuroboPlanner  # from curobo_overlay/, copied onto cuRobo by the Dockerfile

PORT = int(os.environ.get("CUROBO_PORT", "8004"))
ROBOT_TYPE = os.environ.get("CUROBO_ROBOT_TYPE", "franka_robotiq")

app = FastAPI()
planner = CuroboPlanner(robot_type=ROBOT_TYPE)


# ---------- request models ----------
class PlanRequest(BaseModel):
    goal_pose: list      # length-7 [x,y,z,qw,qx,qy,qz] or 4x4 list
    start_state: list    # length-7 joint state
    constrained: bool = False


class PlanGraspRequest(BaseModel):
    goal_pose: list
    start_state: list
    constrained: bool = False


# ---------- endpoints ----------
@app.post("/plan")
def plan(req: PlanRequest):
    goal_pose = np.array(req.goal_pose)
    start_state = np.array(req.start_state)
    success, traj, dt = planner.plan(goal_pose, start_state, constrained=req.constrained)
    if not success:
        print("Planning result: FAILED")
        return {"success": False}
    print("Planning result:", success, len(traj.position.tolist()), dt)
    return {"success": success.item(), "trajectory": traj.position.tolist(), "dt": dt}


@app.post("/plan_grasp")
def plan_grasp(req: PlanGraspRequest):
    goal_pose = np.array(req.goal_pose)
    start_state = np.array(req.start_state)
    success, res = planner.plan_grasp(goal_pose, start_state)
    print("Grasp planning result:", success)
    if not success:
        return {"success": False}
    return {
        "success": success.item(),
        "trajectory": res.grasp_interpolated_trajectory.position.tolist(),
        "dt": res.grasp_interpolation_dt,
    }


@app.post("/fuse_observation")
def fuse_observation(views: UploadFile):
    try:
        with np.load(io.BytesIO(views.file.read()), allow_pickle=False) as data:
            dataset = [{"depths": depth, "K": K, "camtoworld": pose}
                       for depth, K, pose in zip(data["depths"], data["intrinsics"], data["camera_poses"])]
        planner.fuse_observation(dataset)
        return {"success": True, "views": len(dataset)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/clean_map")
def clean_map():
    planner.clean_map()
    return {"success": True}


@app.post("/toggle_collision_for_grasp")
def toggle_collision_for_grasp(flag: bool):
    planner.toggle_collision_for_grasp(flag)
    return {"success": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
