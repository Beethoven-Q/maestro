# Hardware

Maestro runs on the [DROID](https://github.com/droid-dataset/droid) platform. If your robot is
set up for DROID, it is set up for Maestro.

## Robot

- A Franka **FR3** or **Panda** arm, driven through DROID's robot server running on a separate
  computer (the "NUC"). Maestro connects to it at `robot.nuc_ip`. cuRobo plans with the
  Panda's joint limits; some FR3 limits are tighter (joints 1, 2, 4, 5 and the lower limit of
  joint 6), so on an FR3 a rare plan can approach a limit and stop the arm.
- A **Robotiq 2F-85** gripper. Maestro is set up for it: the cuRobo service models it, and the
  client places the fingertip centre 0.14 m from the robot flange (`GRIPPER_FLANGE_OFFSET` in
  `maestro/api.py`). Another gripper, including the standard Franka Hand, needs both changed.

## Cameras

- **One ZED wrist camera** (we use a ZED Mini). It does all of the perception: scanning,
  pointing, segmentation, depth and grasping. It needs a USB 3 connection that stays solid
  while the arm moves; a loose or USB 2 cable makes the camera drop out.
- **Two external ZED cameras** (we use ZED 2; one also works). They give the VLM a second view
  of the scene and record the trial video. Our external cameras are mounted upside down, so
  Maestro opens every non-wrist ZED with its image flipped (`FLIP_MODE.ON` in
  `droid/camera_utils/camera_readers/zed_camera.py`). If yours are mounted upright, change
  both occurrences to `FLIP_MODE.OFF` before calibrating.
- Put the serial numbers in `config/maestro.yaml` under `cameras`.

## Calibration

Maestro needs to know where each camera is relative to the robot. Calibrate with DROID's camera
calibration procedure (a ChArUco board; see the DROID documentation), then place the resulting
`calibration_info.json` in `droid/calibration/`. Recalibrate whenever a camera or its mount
moves.

The wrist calibration matters most: every 3D position Maestro computes comes from the wrist
camera. If grasps land consistently too high or too low, check this calibration first.

## Compute

- **Robot workstation:** Ubuntu 22.04 with the ZED SDK. The Maestro client itself needs no GPU,
  but the ZED SDK uses the workstation's NVIDIA GPU.
- **GPU services:** together they hold about 18 GB of GPU memory once loaded (cuRobo about 8,
  FoundationStereo about 7, Grounded-SAM about 2, GraspGen about 1), with extra while they
  work; one 24 GB GPU or more fits all four. They can also be split across machines. In our setup FoundationStereo and GraspGen share the workstation's 16 GB
  laptop GPU, and Grounded-SAM and cuRobo run on a server.
