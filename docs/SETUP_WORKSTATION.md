# Install the client on the robot workstation

The workstation runs Maestro itself: the planning loop, the robot tools and the camera
drivers. The models run as separate GPU services ([SETUP_SERVICES.md](SETUP_SERVICES.md)),
so the client needs no PyTorch or CUDA toolkit.

## 1. What to have first

- Ubuntu 22.04 with the [ZED SDK](https://www.stereolabs.com/developers/release) installed
  for your cameras.
- The Franka arm running DROID's robot server on its control computer, reachable from this
  workstation.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), the Python package manager:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 2. Install

```bash
git clone https://github.com/Beethoven-Q/maestro.git && cd maestro
uv sync                                                # creates .venv with the exact locked packages
uv run python /usr/local/zed/get_python_api.py         # downloads the ZED Python wheel (pyzed)
uv pip install --python .venv pyzed-*.whl              # installs it into Maestro's environment
```

ZED's script ends with a pip error ("pyzed was NOT correctly setup") because uv environments
have no pip; ignore it, the next line installs the wheel.

`uv sync` keeps the environment exactly in line with `uv.lock`, so running it again removes
pyzed; reinstall the wheel afterwards.

## 3. Configure

```bash
cp .env.example .env                                   # add GEMINI_KEY and/or OPENAI_API_KEY
cp config/maestro.example.yaml config/maestro.yaml
```

In `config/maestro.yaml` set your camera serial numbers, the robot server's address, where
each GPU service runs, and the VLM provider ([CONFIGURATION.md](CONFIGURATION.md)). Put your
camera calibration in `droid/calibration/calibration_info.json` ([HARDWARE.md](HARDWARE.md)).

## 4. Check without moving the robot

```bash
uv run python -m unittest discover -s tests            # unit tests
uv run python scripts/check_camera_depth.py --output /tmp/camera_check --depth
```

The second command opens the wrist camera, saves a stereo pair, sends it to the
FoundationStereo service and prints the depth it gets back. If that works, the camera, the
calibration readout and the depth service are all reachable.

## 5. Run

With the GPU services running:

```bash
uv run python -m maestro.runner -i "put the red block in the bowl"
```

Options: `-c code.py` runs a Python file as the robot code instead of asking the planner (useful
to test tools); `-l runs/logs_all_tasks` gives the planner earlier trials to learn from.
Press Ctrl+C to stop a trial; the video is saved and you can start another.
