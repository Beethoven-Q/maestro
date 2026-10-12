# FoundationStereo depth service

Stereo → metric depth with [FoundationStereo](https://github.com/NVlabs/FoundationStereo).
Flask on `:5005`.

- `POST /depth`: multipart files `left` and `right` (a rectified PNG pair) and form fields `fx`
  (left focal length, pixels) and `baseline` (meters) → float32 `.npy` depth in meters.
- `GET /health` → ready once the model has loaded.

The image pins FoundationStereo's commit, the Python packages in `requirements.txt` and the
checkpoint (`model_best_bp2.pth`, checked by SHA-256, with its `cfg.yaml` from this directory).
Run it with `docker compose up -d --build foundation_stereo` from the repository root; see
[SETUP_SERVICES.md](../../docs/SETUP_SERVICES.md).
