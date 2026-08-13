# Lab 4: TODO

## Canonical UR5e + Robotiq Path
- [x] Step R1: Replace the old Lab 4 robot/collision baseline with Menagerie UR5e + mounted Robotiq 2F-85
- [x] Step R2: Align planning-time collision truth with the executed MuJoCo geometry
- [x] Step R3: Re-validate RRT / RRT* on the canonical stack
- [x] Step R4: Re-validate path shortcutting and time parameterization
- [x] Step R5: Re-validate executed tracking on the canonical stack
- [x] Step R6: Record a Lab 4 validation video on the canonical stack
- [x] Step R7: Update README and task docs to the signed-off final state

## Video Production
- [x] Step V1: Shared `tools/video_producer.py` three-phase pipeline
- [x] Step V2: Slalom demo with multi-waypoint RRT* planner, metrics, and video
- [x] Step V3: Consolidated all output into `media/`

## Slalom Redesign (2026-03-24)
- [x] Step S1: Replace obstacles with 4 staggered tabletop boxes (10x10x20 cm)
- [x] Step S2: Add 9-waypoint slalom path at z=0.56 with gap-midpoints
- [x] Step S3: Rewrite capstone_demo.py as multi-segment RRT* weaving demo
- [x] Step S4: Rewrite record_lab4_demo.py and record_lab4_validation.py for slalom
- [x] Step S5: Delete slalom_demo.py and generate_lab4_demo.py (absorbed into capstone)
- [x] Step S6: Update tests — all 44 pass with new obstacle layout
- [x] Step S7: Update architecture and task docs

## Doc / Artifact Cleanup (2026-08-13)
- [x] Step D1: Re-ran the full suite — `45 passed` (no failures; `TestVisualization` skips only when `mpl_toolkits.mplot3d` is missing)
- [x] Step D2: Replaced the pre-redesign README "Key Results" table with the 2026-03-24 slalom numbers from `tasks/PLAN.md`; kept the blocked-path numbers behind a clearly labelled "Historical" block
- [x] Step D3: Deleted `media/slalom_metrics.json` as unreproducible — see note below
- [x] Step D4: Documented `record_lab4_demo.py` and its three video outputs (`lab4_metrics.mp4`, `lab4_simulation.mp4`, `lab4_demo.mp4`) in the README module table and media section
- [x] Step D5: Wrote `blog/lab4_blog_post.md` (LAB_04 success criterion "Blog post published")

### Note on the removed `media/slalom_metrics.json`
The file described a "Slalom Through Obstacles (round trip)" scenario with 17
waypoints, a 29.73 s trajectory and `minimum_obstacle_clearance_m: 0.0`. No
current script emits it: the key `minimum_obstacle_clearance` appears in no `.py`
file in the lab, and `capstone_demo.py` / `record_lab4_demo.py` write no JSON at
all. It was produced by `slalom_demo.py` / `generate_lab4_demo.py`, both deleted
in Step S5 of the slalom redesign. Its contents also contradicted the validated
`tasks/PLAN.md` numbers (forward-only run: 24 path waypoints, 15.22 s duration,
0.034 m minimum clearance). Removed rather than regenerated because there is no
current code path that can reproduce it. The slalom `*.png` plots from the same
generation are retained as static artifacts.

## Current Focus
> Lab 4 complete. Slalom capstone validated — arm weaves through 4 obstacles.
> Docs, metrics and media inventory reconciled with the post-redesign pipeline.

## Blockers
> None.
