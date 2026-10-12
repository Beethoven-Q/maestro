# Robot tools

The planner VLM controls the robot by writing Python that calls two objects: `robot` (the
skills below) and `image_editor` (drawing on the feedback image). The VLM learns these tools
from their descriptions in [`maestro/prompts/maestro_prompt.txt`](../maestro/prompts/maestro_prompt.txt);
they are implemented in [`maestro/api.py`](../maestro/api.py).

Positions are `[x, y, z]` in meters in the robot base frame: x points forward, away from the
robot, y to the robot's left, z up. Orientations are `[roll, pitch, yaw]` in radians.
`[pi, 0, 0]` points the gripper straight down; `[0, pi/2, 0]` points it horizontally forward.
A position always means the centre between the gripper's fingertips.

## Finding objects

All perception uses the wrist camera. At the start of a task the robot scans the table from 8
viewpoints, and later calls reuse that scan.

| Tool | What it does | Speed |
|---|---|---|
| `locate_objects(labels)` | Asks the VLM to point at each object across all scan views at once, then turns each point into a 3D position. Returns `[{'label', '2d_position', '3d_position'}]` for the objects it found. | fast, first choice |
| `locate_object_by_segmentation(name, z_offset=0.0)` | Segments the object with Grounded-SAM and returns the 3D position of its bounding-box centre. | fast |
| `locate_object_part(name, part_description)` | Places numbered points on the object's segmentation mask, asks a VLM which ones match the description (for example "the handle of the knife"), and returns their 3D positions. | slower, most precise |
| `detect_bounding_box(name)` | The object's bounding box and detection score in the current wrist view. | fast |
| `refresh_scan()` | Scans the table again, for example after objects moved. | ~40 s |

## Grasping and moving

| Tool | What it does |
|---|---|
| `propose_grasps(name)` | Looks at the object from up to three viewpoints, fuses the views into a point cloud, and returns GraspGen's grasp candidates as `(position, orientation)` pairs. |
| `execute_grasp(grasps)` | Plans a collision-free approach to the best reachable candidate with cuRobo, moves there and closes the gripper. |
| `move_to_pose(position, orientation)` | Moves the gripper to a pose along a collision-free path. Returns `False` if no path was found. |
| `open_gripper()`, `close_gripper()` | Open or close the gripper. |
| `get_robot_state()` | The gripper position and orientation, and `gripper_fraction_closed` (0 open, 1 fully closed; in between usually means something is held). |
| `reset()` | Moves the arm back to its reset pose and opens the gripper. |

Collision-free planning uses a 3D map built from the wrist scan's depth images.

## Geometry

Exact helpers, so the VLM computes poses instead of estimating them.

| Tool | Returns |
|---|---|
| `measure_vector(p1, p2)` | the vector from `p1` to `p2` |
| `measure_distance(p1, p2)` | the distance between two points |
| `measure_relative_rotation(v1, v2)` | the rotation `[roll, pitch, yaw]` that turns direction `v1` into `v2` |
| `rotate_by(orientation, rotation)` | `orientation` rotated by `rotation` |
| `euler_to_rotation_matrix(euler)`, `rotation_matrix_to_euler(matrix)` | conversions |

## Drawing on the feedback image

After each step the VLM receives an image from the wrist and external cameras. These calls add
marks to that image so the VLM can check a result visually:

| Tool | Draws |
|---|---|
| `image_editor.draw_point(position)` | a 3D point |
| `image_editor.draw_pose(position, orientation)` | the axes of a pose; two poses align if their axes look parallel |

Points found by `locate_objects` are drawn automatically, with their labels.

## Adding a tool

1. Add a method to `RealFrankaRobotApi` in `maestro/api.py`. Start it with
   `self._log_timestamp("FUNCTION_CALL", ...)` so the call appears in the trial's timestamp log.
2. Describe it in the API section of `maestro/prompts/maestro_prompt.txt`, with its arguments,
   return value and when to use it. The VLM only knows what the prompt tells it.
3. If it needs a model, add a service under `servers/` that takes all its inputs in the request
   (see [ARCHITECTURE.md](ARCHITECTURE.md)) and a client function in `maestro/perception_utils.py`.
