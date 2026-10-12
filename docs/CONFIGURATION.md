# Configuration

Maestro reads two files from the repository root. Both are ignored by git.

| File | Holds | Start from |
|---|---|---|
| `.env` | API keys | `cp .env.example .env` |
| `config/maestro.yaml` | everything else | `cp config/maestro.example.yaml config/maestro.yaml` |

Values like `${MAESTRO_HOME}` are filled in from environment variables; `MAESTRO_HOME` is the
repository folder unless you set it.

## `.env`

| Variable | Needed for |
|---|---|
| `GEMINI_KEY` | `vlm.provider: gemini` |
| `OPENAI_API_KEY` | part-point selection with Gemini, and everything with `vlm.provider: astra` |

## `config/maestro.yaml`

### `vlm`: which models think

```yaml
vlm:
  provider: gemini        # or astra
  gemini:
    planner: gemini-2.5-pro                 # writes the plan and code
    temperature: 0.1
    pointing: gemini-3.1-flash-image-preview # locate_objects
    part_selection: o3-2025-04-16           # locate_object_part (OpenAI)
    api_version: v1alpha
  astra:
    model: gpt-6-astra                      # all three roles
    reasoning_effort: medium                # low, medium, high, xhigh or max
```

### `services`: where the GPU services run

One `host` and `port` per service. Use `localhost` for services on the workstation itself.

```yaml
services:
  foundation_stereo: {host: localhost, port: 5005}
  groundedsam:       {host: localhost, port: 8005}
  graspgen:          {host: localhost, port: 8006}
  curobo:            {host: localhost, port: 8004}
```

### `paths`

`data_dir` is where trial logs, videos and perception data are written (default `runs/`).

### `robot`

| Key | Meaning |
|---|---|
| `model` | `fr3` or `panda` |
| `use_motionplanner` | `true` plans collision-free paths with cuRobo; `false` moves straight to each pose without collision checking |
| `nuc_ip` | address of the computer running DROID's robot server |
| `robot_ip`, `laptop_ip` | the Franka controller and this workstation on the robot network |
| `serial_number` | optional, stored in recorded trajectories |

### `cameras`

ZED serial numbers (printed on the camera, or shown by `ZED_Explorer`).

| Key | Camera |
|---|---|
| `hand_camera_id` | the wrist camera; required, used for all perception |
| `varied_camera_1_id`, `varied_camera_2_id` | the two external cameras (one also works). The trial video is recorded from camera 2, or camera 1 if there is no camera 2, and the feedback image the VLM sees includes an external view |

## Calibration

The camera-to-robot calibration lives in `droid/calibration/calibration_info.json`; see
[HARDWARE.md](HARDWARE.md). The ZED's own lens intrinsics are read from the camera at runtime.
