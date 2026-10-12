"""Maestro GroundedSAM segmentation server (FastAPI).

Open-vocabulary segmentation: GroundingDINO (text -> boxes) + SAM (boxes -> masks).
POST /segment with an image file `image` and a form field `prompt`. The response body is an
.npz archive with, per detection: `masks` (N, H, W bool), `boxes` (N, 4 pixel x0, y0, x1, y1),
`scores` (N,) and `labels` (N,) — empty arrays when nothing is found.

Configured via environment variables:
  GROUNDEDSAM_PORT                 listen port (default 8005)
  GROUNDINGDINO_CKPT               groundingdino_swint_ogc.pth (required)
  SAM_CKPT                         sam_vit_h_4b8939.pth (required)
  BOX_THRESHOLD / TEXT_THRESHOLD   detection thresholds (defaults 0.3 / 0.25)

Requires the `groundingdino` and `segment_anything` packages of Grounded-Segment-Anything.
"""
import os
from io import BytesIO

import groundingdino
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import Response
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

app = FastAPI()

# ---------------- config ----------------
PORT = int(os.environ.get("GROUNDEDSAM_PORT", "8005"))
CONFIG_FILE = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")
GROUNDED_CHECKPOINT = os.environ["GROUNDINGDINO_CKPT"]
SAM_CHECKPOINT = os.environ["SAM_CKPT"]
BOX_THRESHOLD = float(os.environ.get("BOX_THRESHOLD", "0.3"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.25"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- helpers ----------------
def load_image_from_bytes(image_bytes):
    image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(config_path, ckpt_path, device):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("GroundingDINO load result:", load_res)
    model.eval()
    return model.to(device)


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenized = model.tokenizer(caption)
    phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer)
               for logit in logits_filt]
    return boxes_filt, logits_filt.max(dim=1)[0], phrases


# ---------------- model cache ----------------
grounding_dino_model = None
sam_predictor = None


@app.on_event("startup")
def load_all_models():
    global grounding_dino_model, sam_predictor
    grounding_dino_model = load_model(CONFIG_FILE, GROUNDED_CHECKPOINT, DEVICE)
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    sam_predictor = SamPredictor(sam)
    print("All models loaded once at startup.")


# ---------------- API ----------------
@app.post("/segment")
async def segment(image: UploadFile, prompt: str = Form(...)):
    image_pil, image_tensor = load_image_from_bytes(await image.read())
    boxes, scores, labels = get_grounding_output(
        grounding_dino_model, image_tensor, prompt, BOX_THRESHOLD, TEXT_THRESHOLD, device=DEVICE
    )
    image_np = np.array(image_pil)
    H, W = image_np.shape[:2]
    # GroundingDINO boxes are normalized (cx, cy, w, h); SAM takes pixel (x0, y0, x1, y1).
    boxes = boxes * torch.Tensor([W, H, W, H])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    masks = np.zeros((0, H, W), dtype=bool)
    if len(boxes):
        sam_predictor.set_image(image_np)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2]).to(DEVICE)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False
        )
        masks = masks[:, 0].cpu().numpy()
    print(f"Segmentation of '{prompt}': {len(masks)} masks found.")

    buffer = BytesIO()
    np.savez_compressed(buffer, masks=masks, boxes=boxes.numpy().astype(np.float32),
                        scores=scores.numpy().astype(np.float32), labels=np.array(labels, dtype=str))
    return Response(buffer.getvalue(), media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
