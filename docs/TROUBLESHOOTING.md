# Troubleshooting

### "Set cameras.hand_camera_id in config/maestro.yaml"
Maestro needs the wrist camera's serial number. Find it with `ZED_Explorer` (or on the camera)
and put it in `config/maestro.yaml` under `cameras`.

### "Wrist camera <serial> not found"
The wrist ZED is not connected, or `cameras.hand_camera_id` has a typo. Check that
`lsusb -t` shows the ZED-M as a Video device at 5000M (USB 3), and that the serial matches the
one `ZED_Explorer` shows.

### "Missing wrist calibration for <serial>_left"
The wrist camera has not been calibrated yet; calibrate it ([HARDWARE.md](HARDWARE.md)).

### A service does not answer
```bash
curl http://<host>:5005/health                              # FoundationStereo
curl -o /dev/null -w '%{http_code}\n' http://<host>:8005/docs   # Grounded-SAM, prints 200
curl http://<host>:8006/health                              # GraspGen
curl -X POST http://<host>:8004/clean_map                   # cuRobo
```
Check that the host and port in `config/maestro.yaml` match, that the container is up
(`docker compose ps`, `docker compose logs <service>`), and that no firewall blocks the port.
A service needs a minute or two after starting to load its model.

### "Motion planning failed"
cuRobo found no collision-free path, usually because the target pose would collide with
something in the 3D map, sometimes because of a joint limit. This is a normal outcome the
planner VLM is told how to handle (back off, change orientation, lift higher). If it happens
for every move, look at the cuRobo log: each fusion prints how many depth views it received,
and a wrong camera calibration puts obstacles in the wrong place.

### Objects are found in the wrong place, or grasps land too high or too low
3D positions come from the wrist ZED's own depth and the wrist camera calibration. Recalibrate
the wrist camera ([HARDWARE.md](HARDWARE.md)), and look at its depth with the ZED SDK's
`ZED_Depth_Viewer`.

### The wrist camera stops during a trial
Maestro retries a few times, reopens the camera, and stops with "stopped delivering frames"
if it does not recover. This is almost always the USB connection: use a USB 3 cable that is
fixed along the arm so it does not pull or bend at the wrist, then restart Maestro.

### API key errors
Set `GEMINI_KEY` and/or `OPENAI_API_KEY` in `.env`. `vlm.provider: astra` needs only the OpenAI
key; `gemini` needs both (o3 selects part points).

### Docker fills the system disk while building
Docker stores images under `/var/lib/docker` by default. Point its `data-root` at a larger disk
in `/etc/docker/daemon.json` before building. With Docker 29 and later, images go to
containerd's store instead unless you also set `"features": {"containerd-snapshotter": false}`.

### Viewing grasps
The GraspGen service shows each request's point cloud and grasps at
`http://<graspgen-host>:7000/static/`.
