# Lab 2 Media

Rendered artifacts for the UR5e lab. Both files show the C3 capstone: the arm
tracing the edges of a cube in Cartesian space.

| File | Format | Size | Details | Produced by |
|---|---|---|---|---|
| `c3_draw_cube.mp4` | H.264 MP4 | ~1.3 MB | 1280x720, 30 fps, 23.6 s | `python3 lab-2-Ur5e-robotics-lab/src/c3_record_video.py` |
| `c3_draw_cube.gif` | GIF | ~6.4 MB | 512x288, 12 fps, 21.2 s (255 frames) | re-encoded from the MP4 render for inline README embedding |

`c3_draw_cube.gif` is the image embedded at the top of the lab
[README](../README.md). It is a downscaled, palette-reduced copy of the same
run as the MP4 — use the MP4 when you want full resolution.

Both are rendered headlessly through MuJoCo offscreen rendering
(`MUJOCO_GL=egl`), so no display is needed to regenerate them.

## Not recorded as video

The four C1 pipeline demos were never rendered to video; they exist only as
numeric logs. Each CSV holds the per-step time, phase, end-effector error,
manipulability, torque norm, and collision flag:

| Demo | Log |
|---|---|
| Pick and place | [`../docs/c1_pick_place_log.csv`](../docs/c1_pick_place_log.csv) |
| Multi-waypoint tour | [`../docs/c1_multi_waypoint_log.csv`](../docs/c1_multi_waypoint_log.csv) |
| Circle tracking | [`../docs/c1_circle_log.csv`](../docs/c1_circle_log.csv) |
| Singularity stress test | [`../docs/c1_singularity_log.csv`](../docs/c1_singularity_log.csv) |

Summary metrics across all four runs are in
[`../docs/c1_metrics_dashboard.csv`](../docs/c1_metrics_dashboard.csv), and the
write-up is [`../docs/c1_full_pipeline.md`](../docs/c1_full_pipeline.md).

`python3 lab-2-Ur5e-robotics-lab/src/c1_pick_and_place.py` regenerates the
pick-and-place and circle logs (plus `../docs/c1_metrics.csv`). The
multi-waypoint, singularity, and dashboard CSVs are archived output from an
earlier revision of that script and are not reproduced by the current version.
