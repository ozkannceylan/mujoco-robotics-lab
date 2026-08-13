---
title: "50 Lessons from Building a Humanoid VLA System"
date: "2025-03-01"
description: Engineering lessons learned while building a Vision-Language-Action system for the Unitree G1 humanoid robot in MuJoCo.
tags: [Robotics, MuJoCo, Imitation Learning, ROS 2]
type: blog
website: /blog/50-lessons-humanoid-vla
---

Building a Vision-Language-Action system for the Unitree G1 humanoid robot taught me more about robotics engineering than any course could. Here are the key lessons, condensed from 10 weeks of development across 6 project phases.

## Key Insights

- **Torque actuators are not position actuators** - Need PD controller: τ = Kp(q_des - q) - Kd * q̇ + τ_gravity
- **Gravity compensation matters from day one**
- **Kinematic IK is fragile** - Always validate generated demos before training
- **Action chunking is the key insight** - Predicting 20 timesteps at once dramatically improves temporal consistency
- **Freeze early ResNet layers** - Prevents overfitting on small robotics datasets
- **Temporal ensembling smooths execution**
