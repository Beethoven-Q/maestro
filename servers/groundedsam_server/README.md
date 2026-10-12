# GroundedSAM segmentation service

Open-vocabulary segmentation with GroundingDINO (text → boxes) and SAM ViT-H (boxes → masks),
from [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).
FastAPI on `:8005`.

- `POST /segment`: file `image`, form `prompt` → `.npz` with `masks` (N×H×W bool), `boxes`
  (N×4 pixel x0, y0, x1, y1), `scores` (N) and `labels` (N); empty arrays when nothing is found.

The image pins the upstream commit (plus `groundingdino-scalar-type.patch`, which lets its CUDA
op build with current PyTorch), the packages in `requirements.txt`, and the checkpoints and BERT
text encoder by hash. `BOX_THRESHOLD` / `TEXT_THRESHOLD` (defaults 0.3 / 0.25) tune detection.
Run it with `docker compose up -d --build groundedsam` from the repository root; see
[SETUP_SERVICES.md](../../docs/SETUP_SERVICES.md).
