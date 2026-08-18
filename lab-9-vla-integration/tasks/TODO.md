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

## M1 — Demonstration dataset — ✅ DONE (2026-08-18), GATE PASSED 5/5
- [x] 1.1 `collect_demos.py` — 120 episodes (60 seeds x 2 objects), phase-sliced
- [x] 1.2 manifest + normalisation statistics fitted on the train split only
- [x] 1.3 `dataset.py` — chunked windows, pad masks, **seed-level** split
- [x] Gate: 120 demos/task (want >= 50); 120/120 episodes succeeded; no seed
      leakage (48 train / 12 val); 12,180 frames, 244 MB, 38.4 min on 4 cores
- [x] Evidence: `media/m1_dataset_grid.png`, `data/manifest.json`
- [x] Lessons: L-M1-a … L-M1-d

## M2 — Model — ✅ DONE (2026-08-18), GATE PASSED 7/7
- [x] 2.1 `text_encoder.py` + instruction bank baked into the checkpoint
- [x] 2.2 `act_policy.py` — 15.75 M params (12.96 M trainable), two cameras,
      token count derived from the image size, both action heads
- [x] Gate: shapes, determinism, instruction sensitivity, meaning-vs-paraphrase
      margin 0.111, overfit 0.250 x the constant-predictor baseline, checkpoint
      round-trip exact
- [x] Evidence: `media/m2_model.json`, `tests/test_model.py`
- [x] Lessons: L-M2-a (a failed overfit check can be the check's fault — 3e-3
      destabilises this transformer), L-M2-b

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
> **M3 — training**, then M4's closed-loop evaluation. M0–M2 all passed.
>
> Training is the wall-clock pole: ~4 min/epoch on 4 CPU cores, so the primary
> run (task head, text conditioning, 30 epochs) is ~2 hours. The checkpoint is
> saved on every validation improvement, so evaluation can start against the
> current best at any point.
>
> The gate is **not** "the loss went down" — a policy that has learned the
> dataset's average pose also produces a smooth, falling curve. It is whether
> validation error beats the predict-the-mean baseline by a clear margin on
> scene seeds never trained on.

## Blockers
> None. torch 2.13.0+cpu and torchvision 0.28.0+cpu installed; HuggingFace
> reachable (CLIP weights download at M2, then baked into the checkpoint).
> No CUDA device — planned around, see PLAN § deviation 2.
