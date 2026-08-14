# Lab 8 — Lessons

> Live journal. Log bugs/fixes/insights AS THEY HAPPEN (Symptom / Root cause / Fix /
> Takeaway). Seeded at kickoff with the inherited constraints this lab exists to answer.

## Inherited at kickoff (2026-08-14)

### I-1: Position actuators cannot walk (Lab 7's terminal finding)
Menagerie G1 `<position>` servos track quasi-static references only; dynamic ZMP
replay fails structurally (Lab 7 M3e, 6 attempts — IK converges, PD replay diverges).
Lab 8's entire M0 exists because of this: torque actuators + RNEA inverse dynamics
are the unblock. If M3 walking fails here too, the diagnosis to revisit is the
*controller*, not the actuator model — that hypothesis has already been spent.

### I-2: The analytical model must model the simulated body (Lab 5 L-6.1c)
Build the Pinocchio model from the SAME MJCF MuJoCo steps (`g1_torque.xml`), and
gate M0 on g(q)/M(q) cross-validation — not just FK.

### I-3: Raw diagonal PD chatters on small-inertia joints at 1 kHz (Lab 5 L-6.1b)
Kd·dt/I > 2 is discretely unstable. Shape gains through M(q) from the start; G1
wrist/ankle joints are exactly the risk class.

### I-4: State machines need convergence gates and post-conditions (Lab 5 L-6.1e/f)
Transitions gate on measured convergence with logged residuals; the capstone asserts
the payload's final pose. A run must not be able to claim success silently.

### I-5: Evidence discipline (Lab 7 cleanup lesson)
Every `media/` file has exactly one producing script writing exactly that name;
delete outputs of deleted pipelines in the same commit.

## Session log

_(empty — begins with M0)_
