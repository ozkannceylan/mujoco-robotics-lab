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

## M3 — Training — ✅ DONE (2026-08-18), GATE PASSED 5/5
- [x] 3.1 `train.py` — masked L1, baseline-relative validation in raw units
- [x] 3.2 primary policy: 24 epochs, 110 min, val 0.0107 vs 0.0940 baseline
      (0.11x), hand 4.1 mm, gait 0.054, grasp 0.013
- [x] Two labelling bugs found and fixed by *relabelling* (L-M3-a, L-M3-b), plus
      the chunk-head hedging fix (L-M3-d)
- [ ] 3.3 ablations — **not run**. `task_id` conditioning is superseded by the
      direct instruction-sensitivity measurement; the `joint` head is recorded as
      unmeasured (see M5).
- [x] Evidence: `media/m3_training_curves.png`, `media/m3_training.json`

## M4 — Closed-loop evaluation — ❌ DONE (2026-08-18), GATE FAILED
- [x] 4.1 `evaluate.py` — policy at 10 Hz over Lab 8's QP, success on simulated state
- [x] 4.2 seen + position-randomised + held-out paraphrases (36 episodes)
- [x] 4.3 instruction contrast as a **paired** test over the seen episodes — the
      scene is identical for both instructions, so a separate "swap" condition
      would have been the same measurement twice
- [x] Gate result: 25 % overall in every condition; walk 50 %, pick 0 %;
      commanded stopping separation 0.159 m, produced **0.000 m**
- [x] Evidence: `media/m4_success_rates.png`, `media/m4_episodes.csv`,
      `media/m4_summary.json`
- [x] Lessons: L-M4-a (the demonstrations do not force language use),
      L-M4-b (the reach plateaus 12 mm short of the grasp gate)

## M5 — Capstone + inference profiling — ⚠️ DONE (2026-08-18), inference PASS / task FAIL
- [x] 5.1 `capstone_demo.py` — free-form sentence in, no task index on the path
- [x] 5.2 latency: **37.0 Hz** float32, **38.4 Hz** dynamically quantised
- [x] Gate: inference > 10 Hz PASS; episode success FAIL (walked correctly,
      never grasped); no fall, 51 N.m of a 139 N.m limit
- [x] Evidence: `media/m5_capstone.mp4`, `media/m5_capstone.json`

## M6 — Documentation & blog — ✅ DONE (2026-08-18)
- [x] `docs/ARCHITECTURE.md`, `docs/CODE_WALKTHROUGH.md`
- [x] `docs-turkish/ARCHITECTURE_TR.md`
- [x] `blog/lab9_vla_integration.md` — the nine-lab arc, written in-milestone
- [x] Lab README with every milestone's gate table, including the failed ones
- [x] Root README / MASTER_PLAN / CLAUDE.md / tasks/todo.md / plan/LAB_09.md

## Current Focus
> **None — Lab 9 is closed at its measured scope (2026-08-18).**
>
> What passed: the scene and contract (M0), the demonstration set (M1), the model
> (M2), training (M3), and the capstone's inference budget — 37 Hz on four CPU
> cores against a 10 Hz bar.
>
> What failed, and why, both measured rather than guessed:
> - **The policy ignores its instruction** (0.3 mm difference in the hand target
>   between the two commands). The two-object scene makes language necessary in
>   principle; the expert's own behaviour makes it redundant in the data, because
>   it walks until the named object is the one in front of it. Fixing this is a
>   re-collection with a target-independent stopping point — L-M4-a.
> - **The reach plateaus 12 mm short of the grasp gate** and holds there. Ordinary
>   compounding error terminating in the same absorbing state as L-M3-b — L-M4-b.
>
> Not run and not dropped: the joint-head ablation promised in `tasks/PLAN.md`
> deviation 3. The code path exists and is documented; the run was not made.

## Blockers
> None.
