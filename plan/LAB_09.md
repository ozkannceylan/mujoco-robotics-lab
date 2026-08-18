# Lab 9: VLA Integration

> **Status:** ⚠️ Closed 2026-08-18 at measured scope — M0–M3 and M5's inference
> gate passed; **M4 and M5's task gate failed, with the cause measured**. The
> policy trains cleanly (val 0.11x the mean baseline, 4.1 mm hand error), stops
> within 1 mm of the right place, runs at 37 Hz on CPU — and ignores its
> instruction (0.3 mm difference in the hand target between the two commands),
> because the expert's own behaviour makes language redundant in the data.
> Scope cut to 2 tasks (`walk`, `pick`) from 3–5: `carry` 1/12 and `place` 5/10
> for the *expert*. Results: `lab-9-vla-integration/README.md`.  
> **Prerequisites:** Lab 8 (whole-body loco-manipulation), humanoid_vla project  
> **Platform:** Unitree G1 on MuJoCo + Cloud GPU for training  
> **Capstone Demo:** "Pick up the red cup" — language command → autonomous execution

---

## Objectives

1. Set up vision pipeline: egocentric camera → observation for policy
2. Collect demonstration data using the controllers from Labs 3–8
3. Train an ACT policy conditioned on language instructions
4. Deploy the trained policy on the G1 in MuJoCo — end-to-end from language to action

---

## Why This Lab Matters

This is the destination. Every preceding lab built a piece of the puzzle: kinematics for understanding poses, dynamics for safe interaction, planning for obstacle avoidance, manipulation for doing useful work, locomotion for mobility, and whole-body control for combining them. Now, a learned policy replaces the hand-coded pipeline. The VLA takes camera images + language → joint actions, collapsing the entire stack into a single neural network. Understanding the manual pipeline (Labs 1–8) is what lets you debug, evaluate, and improve the learned one.

---

## Theory Scope

- Vision-Language-Action (VLA) architecture: vision encoder + language encoder + action decoder
- Action Chunking with Transformers (ACT): predict sequences of actions, not single steps
- Imitation learning: behavior cloning from demonstrations
- Domain randomization: texture, lighting, object pose variation for robustness
- Language conditioning: embedding task instructions into the policy

---

## Architecture

```
                    "Pick up the red cup"
                            │
                            ▼
                  ┌──────────────────┐
                  │ Language Encoder  │
                  │ (frozen LLM or   │
                  │  CLIP text enc)  │
                  └────────┬─────────┘
                           │ language embedding
                           ▼
Camera (30Hz) ──→ ┌──────────────────┐
                  │   VLA Policy      │
                  │   (ACT model)     │
                  │                   │
                  │   vision + lang   │
                  │   → action chunk  │
                  └────────┬─────────┘
                           │ joint actions (chunk of ~10 steps)
                           ▼
                  ┌──────────────────┐
                  │  MuJoCo G1       │
                  │  Execute actions  │
                  │  Step physics     │
                  │  Render camera    │
                  └──────────────────┘
                           │
                           ▼
                  Next camera frame → loop
```

---

## Implementation Phases

### Phase 1 — Scene & Data Pipeline
- Create MuJoCo scenes for target tasks (tabletop with objects: cup, box, bottle)
- Set up egocentric camera rendering (wrist cam + head cam)
- Build demonstration collector: use Lab 5/8 controllers to generate expert trajectories
- Implement domain randomization: object colors, positions, lighting, textures
- Target: 50–100 demos per task, 3–5 tasks

### Phase 2 — Model Architecture
- Port / adapt the ACT model from humanoid_vla project
- Add language conditioning (CLIP text encoder → cross-attention in policy)
- Define observation space: camera image(s) + proprioception (joint positions/velocities)
- Define action space: joint position targets for all actuated DOFs

### Phase 3 — Training
- Train on collected demonstrations (cloud GPU: Lambda Labs or RunPod)
- Evaluate: task success rate, trajectory smoothness, generalization to unseen object positions
- Iterate: adjust domain randomization, demo quality, model size
- Target: >70% success rate on seen tasks, >40% on position-randomized variants

### Phase 4 — Deployment & Demo
- Deploy trained policy for real-time inference on MuJoCo G1
- Test with various language commands: "pick up the red cup", "move the box to the left"
- Profile inference speed: must maintain >10Hz for stable control
- Capstone: record end-to-end demo showing language input → autonomous task execution

### Phase 5 — Documentation & Blog
- Write LAB_09.md: VLA architecture, training pipeline, results, failure analysis
- Write blog post: "from manual control to learned autonomy — the VLA journey"
- This blog post can reference the full lab series as a narrative arc

---

## Connection to humanoid_vla Project

This lab merges with and extends the existing `ozkannceylan/humanoid_vla` project:

- **Reuse:** ACT model architecture, MuJoCo G1 setup, IK-based data generation pipeline
- **Extend:** Add language conditioning (humanoid_vla uses fixed tasks), improve domain randomization, integrate controllers from Labs 3–8 as expert demonstrators
- **Upgrade:** Replace the 5-task setup with a broader task set informed by the full lab series

---

## Key Design Decisions for Claude Code

- **IK-based data generation, not teleoperation.** As established in humanoid_vla, use the controllers from Labs 3–8 to generate demonstrations programmatically. This is faster, more consistent, and doesn't require teleoperation hardware.
- **ACT over diffusion policy.** ACT is simpler, faster to train, and already validated in humanoid_vla (~15.6M params). Diffusion policies are an optional extension.
- **Language conditioning can be simple.** CLIP text encoder + cross-attention is sufficient. Don't build a full LLM backbone — that's scope creep.
- **INT8 for local inference.** Training happens on cloud GPU, but inference should run on local RTX 4050 at >10Hz. Quantize the model.
- **Success detection is part of the pipeline.** Define clear success criteria per task (object at target position ± tolerance) and automate evaluation.

---

## Success Criteria

- [ ] 50+ demonstrations collected per task using programmatic expert
- [ ] ACT model trains successfully with language conditioning
- [ ] >70% success rate on training task configurations
- [ ] >40% success rate on position-randomized variants
- [ ] Real-time inference at >10Hz on local hardware (INT8)
- [ ] Capstone demo: "pick up the red cup" works end-to-end
- [ ] LAB_09.md complete
- [ ] Blog post published — narrative arc covering the full lab series

---

## References

- Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT, 2023)
- Brohan et al., "RT-2: Vision-Language-Action Models" (2023)
- humanoid_vla project: architecture and training pipeline
- CLIP: Radford et al., "Learning Transferable Visual Models" (2021)
