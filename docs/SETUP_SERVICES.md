# Start the GPU services

Maestro's four GPU services run as Docker containers defined in `compose.yaml`:

| service | port | does |
|---|---|---|
| `foundation_stereo` | 5005 | stereo depth |
| `groundedsam` | 8005 | finds objects named in text (Grounded-SAM) |
| `graspgen` | 8006 (+7000 viewer) | grasp candidates |
| `curobo` | 8004 | collision-free motion and grasp planning |

Each request carries all of its data, so any service can run on any machine the Maestro
client reaches. Run all four on one GPU machine, or split them when one GPU is too small.
The reference setup runs `foundation_stereo` and `graspgen` on the robot workstation (16 GB
laptop GPU) and `groundedsam` and `curobo` on a server with two RTX A6000s.

## Requirements
- An NVIDIA GPU and driver (CUDA 12.x capable).
- [Docker Engine](https://docs.docker.com/engine/install/) with the Compose plugin, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  configured for Docker (`sudo nvidia-ctk runtime configure --runtime=docker`).
- Disk space: each image is 20 to 26 GB, including its model weights. The images share their
  CUDA base layers, so four on one machine take less than the sum. Docker keeps images under
  `/var/lib/docker` unless you set another `data-root`.

## Build and start
From the repository root on each GPU machine, start the services it should run:
```bash
docker compose up -d --build                           # all four
docker compose up -d --build foundation_stereo graspgen   # or a subset
```
The first build downloads upstream sources at pinned commits and model weights checked by
SHA-256, and compiles the CUDA extensions; cuRobo's nvblox build takes the longest. The
containers restart with the machine until stopped with `docker compose down`.

## Check
```bash
curl http://<host>:5005/health            # FoundationStereo
curl -o /dev/null -w '%{http_code}\n' http://<host>:8005/docs   # Grounded-SAM, prints 200
curl http://<host>:8006/health            # GraspGen
curl -X POST http://<host>:8004/clean_map # cuRobo
docker compose logs -f <service>          # startup and request logs
```
Then set each service's `host` and `port` in the client's `config/maestro.yaml`
([CONFIGURATION.md](CONFIGURATION.md)). The services have no authentication: publish their
ports only on a trusted network.
