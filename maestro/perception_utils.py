"""Clients for the perception services, and helpers for their results.

`stereo_depth` asks the FoundationStereo service for metric depth of a stereo pair,
`segment` asks the GroundedSAM service for masks of a text prompt, `sample_grasps` asks the
GraspGen service for grasps on an object point cloud, and `fuse_observation` sends depth
views to the cuRobo service's collision world.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import PIL.Image as Image
import requests

from maestro.config import service_url
from maestro.camera_geometry import camera_matrix, stereo_geometry


def _png(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=1)  # lossless; fastest to encode
    return buffer.getvalue()


def stereo_depth(left_image, right_image, calibration):
    """Metric depth (meters) of an RGB stereo pair, from the FoundationStereo service."""
    matrix, baseline = stereo_geometry(calibration)
    response = requests.post(
        service_url("foundation_stereo") + "/depth",
        files={"left": ("left.png", _png(left_image), "image/png"),
               "right": ("right.png", _png(right_image), "image/png")},
        data={"fx": float(matrix[0, 0]), "baseline": baseline},
        timeout=(10, 180))
    response.raise_for_status()
    depth = np.load(io.BytesIO(response.content), allow_pickle=False).astype(np.float32)
    if depth.shape != left_image.shape[:2] or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise RuntimeError("FoundationStereo returned invalid depth")
    return depth


def segment(image, prompt):
    """GroundedSAM detections of `prompt` in an RGB image.

    Returns a dict of per-detection arrays: `masks` (N, H, W bool), `boxes` (N, 4 pixel
    x0, y0, x1, y1), `scores` (N,) and `labels` (N,); N is 0 when nothing is found.
    """
    response = requests.post(
        service_url("groundedsam") + "/segment",
        files={"image": ("image.png", _png(image), "image/png")},
        data={"prompt": prompt},
        timeout=(10, 180))
    response.raise_for_status()
    with np.load(io.BytesIO(response.content), allow_pickle=False) as result:
        return {key: result[key] for key in ("masks", "boxes", "scores", "labels")}


def draw_segmentation(image, detections):
    """RGB image with each detection's mask, box and label drawn on it."""
    canvas = image.copy()
    colors = [(255, 64, 64), (64, 160, 255), (64, 200, 64), (255, 200, 0), (200, 64, 255)]
    for i, (mask, box, score, label) in enumerate(zip(*(detections[k] for k in ("masks", "boxes", "scores", "labels")))):
        color = colors[i % len(colors)]
        canvas[mask] = (0.5 * canvas[mask] + 0.5 * np.array(color)).astype(np.uint8)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, f"{label} {score:.2f}", (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


def sample_grasps(points, colors):
    """GraspGen grasp poses (M, 4, 4, world frame) for an object's world-frame point cloud."""
    buffer = io.BytesIO()
    np.savez(buffer, points=np.asarray(points, dtype=float), colors=np.asarray(colors, dtype=np.uint8))
    response = requests.post(service_url("graspgen") + "/grasps",
                             files={"cloud": ("cloud.npz", buffer.getvalue(), "application/octet-stream")},
                             timeout=(10, 180))
    response.raise_for_status()
    with np.load(io.BytesIO(response.content), allow_pickle=False) as result:
        return result["grasps"]


def gripper_mask(depths, near=0.25, min_views=3, margin_px=7):
    """Pixels the gripper covers in a wrist-camera scan.

    The gripper is fixed in the wrist image, so pixels that read nearer than `near`
    meters in at least `min_views` scan views are the gripper. Stereo depth this close
    to the camera is unreliable, and some views place gripper pixels far away, where
    they would appear as a phantom obstacle; the mask removes them in every view.
    """
    near_views = np.sum([(d > 0) & (d < near) for d in depths], axis=0) >= min_views
    kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)
    mask = cv2.morphologyEx(near_views.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return cv2.dilate(mask, kernel).astype(bool)


def fuse_observation(views):
    """Add depth views to the cuRobo collision world; a planner must not proceed if this fails.

    Each view is a dict with a metric `depth` map, the camera `intrinsics` and its
    camera-to-world `cam_pose`.
    """
    buffer = io.BytesIO()
    np.savez(buffer,
             depths=np.stack([view['depth'] for view in views]).astype(np.float32),
             intrinsics=np.stack([camera_matrix(view['intrinsics']) for view in views]),
             camera_poses=np.stack([np.asarray(view['cam_pose'], dtype=float) for view in views]))
    response = requests.post(service_url('curobo') + '/fuse_observation',
                             files={'views': ('views.npz', buffer.getvalue(), 'application/octet-stream')},
                             timeout=(10, 120))
    response.raise_for_status()
    result = response.json()
    if result.get('success') is not True:
        raise RuntimeError(f"Scene fusion failed: {result.get('error', result)}")
    return result
