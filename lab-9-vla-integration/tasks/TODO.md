# Lab 9: TODO

> Kickoff: 2026-08-17. Rules: gate + evidence in `media/mN_*` per milestone; if a
> gate fails, fix THIS milestone. Brief: `plan/LAB_09.md`. Plan: `tasks/PLAN.md`.

## Pre-M0 (kickoff) — DONE 2026-08-17
- [x] Read `plan/LAB_09.md` fully
- [x] Feasibility probe **before** planning (the numbers that shaped every milestone):
      no CUDA device; 4 cores / 15 GB; MuJoCo EGL render **97 ms/frame and
      resolution-independent** (380 ms with shadows/reflection/skybox);
      ResNet18 fwd+bwd 117 samp/s @128 px; one Lab 8 capstone episode 53.6 s wall
- [x] Inspect `ozkannceylan/humanoid_vla` — found it **already has** CLIP text
      conditioning and spatial vision tokens, so the brief's "extend: add language
      conditioning" is stale. Re-scoped: this lab's contribution is the
      loco-manipulation expert (Labs 3–8), not the language head.
- [x] Create lab folder structure
- [x] Write `tasks/PLAN.md` (M0–M6, four scope deviations documented)
- [x] Write `tasks/ARCHITECTURE.md` (module map, obs/action contract, data flow)
- [x] Create `tasks/TODO.md` + `tasks/LESSONS.md`
- [ ] Update project status boards (deferred to M6, as Lab 8 did)

## M0 — Scene, cameras, observation/action contract — ✅ DONE (2026-08-17), GATE PASSED 6/6
- [x] 0.1 `lab9_common.py` — paths, obs/action constants, instruction vocabulary
- [x] 0.2 `vla_scene.py` — red cup + blue box + drop marker, head + wrist cameras,
      per-seed randomisation, four welds (object x hand)
- [x] 0.3 `observations.py` — obs builder + both action codecs. Base x/y/yaw are
      **excluded** from the state on purpose and a test guards it.
- [x] 0.4 `expert.py` — Lab 8's capstone subclassed, seeded, target-selectable
- [x] 0.5 **Task set cut from 4 tasks to 2, from measurement.** `carry` 1/12 and
      `place` 5/10 against `walk`+`pick` 40/40. Both cuts written up with their
      numbers (L-M0-c, L-M0-e) rather than dropped quietly.
- [x] Gate — all criteria PASS:

      | criterion | result | measured |
      |---|---|---|
      | Expert success rate >= 70 % | PASS | **100 % (40/40)** |
      | Both cameras render at 128 px | PASS | head + wrist |
      | Action round-trip exact | PASS | 5.9e-08 |
      | State matches declared dimension | PASS | 62 |
      | Approach depends on the named object | PASS | 2 or 4 steps |
      | Torques within limits on success | PASS | 92.6 N.m (limit 139) |

- [x] Evidence: `media/m0_scene.png`, `media/m0_expert_rollout.mp4`, `media/m0_gate.json`
- [x] Tests: `tests/test_scene_and_contract.py` — 30 tests
- [x] Lessons: L-M0-a … L-M0-f

## M1 — Demonstration dataset
- [ ] 1.1 `collect_demos.py` (multi-process, phase slicing)
- [ ] 1.2 storage + manifest + train-split-only normalisation stats
- [ ] 1.3 `dataset.py` (chunks, pad mask, seed-level split)
- [ ] Gate: ≥ 50 demos/task, integrity checks, no seed leakage
- [ ] Evidence: `media/m1_dataset_grid.png`

## M2 — Model
- [ ] 2.1 `text_encoder.py` + instruction bank
- [ ] 2.2 `act_policy.py` (derived token count, two cameras, both heads)
- [ ] Gate: param count, shape tests, overfit-one-batch, text changes the output,
      checkpoint round-trip
- [ ] Evidence: printed model table, tests green

## M3 — Training
- [ ] 3.1 `train.py`
- [ ] 3.2 primary policy (`task` head, text conditioning)
- [ ] 3.3 ablations (`task_id` conditioning; `joint` head)
- [ ] Gate: val error beats predict-the-mean by a stated margin; curves recorded
- [ ] Evidence: `media/m3_training_curves.png`

## M4 — Closed-loop evaluation
- [ ] 4.1 `evaluate.py` — policy 10 Hz → Lab 8 QP 1 kHz, success on simulated state
- [ ] 4.2 seen + position-randomised
- [ ] 4.3 instruction-swap test + joint-head ablation
- [ ] Gate: > 70 % seen, > 40 % randomised, instruction swap changes behaviour
- [ ] Evidence: `media/m4_success_rates.png` + per-episode CSV

## M5 — Capstone + inference profiling
- [ ] 5.1 `capstone_demo.py` — free-form language in
- [ ] 5.2 latency: float32 vs dynamic quantisation
- [ ] Gate: no task index on the path; inference > 10 Hz; success on simulated state
- [ ] Evidence: `media/m5_capstone.mp4`

## M6 — Documentation & blog
- [ ] `docs/ARCHITECTURE.md`, `docs/CODE_WALKTHROUGH.md`
- [ ] `docs-turkish/ARCHITECTURE_TR.md`
- [ ] `blog/` — the nine-lab arc
- [ ] Lab README with per-milestone gate tables
- [ ] Root README / MASTER_PLAN / CLAUDE.md / tasks/todo.md / plan/LAB_09.md

## Current Focus
> **M1 — demonstration dataset.** M0 closed 2026-08-17 with a 100 % expert over
> 20 seeds x 2 objects.
>
> The task set is **two** tasks, not the brief's three-to-five: `walk to the
> {object}` and `pick up the {object}`. `carry` and `place` were cut at M0 because
> the *expert* cannot perform them (1/12 and 5/10) — see L-M0-c and L-M0-e. The
> numbers are in the write-up; neither was dropped quietly.
>
> For M1: slice each episode by phase into its two labelled segments, split
> train/val **by seed** (never by frame), and compute normalisation statistics on
> the train split only.

## Blockers
> None. torch 2.13.0+cpu and torchvision 0.28.0+cpu installed; HuggingFace
> reachable (CLIP weights download at M2, then baked into the checkpoint).
> No CUDA device — planned around, see PLAN § deviation 2.
