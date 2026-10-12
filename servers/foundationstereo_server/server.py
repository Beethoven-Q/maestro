"""Maestro FoundationStereo depth server (Flask).

POST /depth with a rectified stereo pair as multipart files `left` and `right` (PNG) and form
fields `fx` (left focal length, pixels) and `baseline` (meters). Responds with the metric
depth map (meters) as a float32 .npy body. The request carries all data, so the service can
run on any machine the client reaches.

Configured via environment variables:
  FOUNDATIONSTEREO_ROOT   FoundationStereo checkout (required)
  FOUNDATIONSTEREO_CKPT   model checkpoint .pth, with its cfg.yaml next to it (required)
  FOUNDATIONSTEREO_HOST   listen address (default 127.0.0.1)
  FOUNDATIONSTEREO_PORT   listen port (default 5005)
"""
import os
import sys

# FoundationStereo's modules (core, Utils) import from its checkout root.
sys.path.insert(0, os.path.realpath(os.environ["FOUNDATIONSTEREO_ROOT"]))

import io

import imageio
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from omegaconf import OmegaConf

from core.utils.utils import InputPadder
from core.foundation_stereo import *  # noqa: F401,F403
from Utils import *  # noqa: F401,F403

app = Flask(__name__)
set_logging_format()
set_seed(0)
torch.autograd.set_grad_enabled(False)

PORT = int(os.environ.get("FOUNDATIONSTEREO_PORT", "5005"))
HOST = os.environ.get("FOUNDATIONSTEREO_HOST", "127.0.0.1")
CKPT_PATH = os.environ.get("FOUNDATIONSTEREO_CKPT")
if not CKPT_PATH:
    raise EnvironmentError("FOUNDATIONSTEREO_CKPT must point to the model checkpoint (.pth)")

# -------------------- load model once --------------------
cfg = OmegaConf.load(os.path.join(os.path.dirname(CKPT_PATH), "cfg.yaml"))
if "vit_size" not in cfg:
    cfg["vit_size"] = "vitl"
cfg.ckpt_dir = CKPT_PATH
args = OmegaConf.create(cfg)

model = FoundationStereo(args)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
del ckpt
model.cuda().eval()
print(f"[FoundationStereo] Model loaded from {CKPT_PATH}")
# ---------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready", "service": "foundation_stereo"})


@app.route("/depth", methods=["POST"])
def depth():
    left = imageio.imread(request.files["left"].read())
    right = imageio.imread(request.files["right"].read())
    fx = np.float32(request.form["fx"])
    baseline = float(request.form["baseline"])
    if left.shape != right.shape or left.ndim != 3 or not (fx > 0 and baseline > 0):
        return jsonify({"error": "expected two RGB images of one size, fx > 0 and baseline > 0"}), 400

    H, W = left.shape[:2]
    img0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.no_grad(), torch.cuda.amp.autocast(True):
        disp = model.forward(img0, img1, iters=32, test_mode=True)
    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W)

    buffer = io.BytesIO()
    np.save(buffer, (fx * baseline / disp).astype(np.float32))
    return Response(buffer.getvalue(), mimetype="application/octet-stream")


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=False)
