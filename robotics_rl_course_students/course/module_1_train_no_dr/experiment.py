# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One experiment for M1 clean tracking and M3 randomized fine-tuning."""
from course.bm_tracker import (
    agent_config, apply_inference_overrides, configure_robot_and_simulator,
    env_config, motion_lib_config, scene_lib_config, terrain_config,
)
from course.bm_tracker import env_config as tracking_env
from course.bm_tracker import configure_robot_and_simulator as configure_clean
from course.bm_tracker import actor_observations
from course.module_3_train_with_dr.randomization import apply_randomization

# M1: leave False. M3: set True and complete the single helper call below.
DOMAIN_RANDOMIZATION = False
RANDOMIZATION_SETTINGS = dict(
    static_friction_range=(0.5, 1.5),  # dimensionless contact friction
    com_offset_m=0.02,                # torso COM displacement per axis, metres
    joint_angle_noise_rad=0.01,       # additive joint-angle observation noise, radians
    push_speed_m_s=0.3,               # horizontal velocity change per axis, m/s
)


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    configure_clean(robot_cfg, simulator_cfg, args)
    if DOMAIN_RANDOMIZATION:
        # TODO M3: call apply_randomization with robot_cfg, simulator_cfg,
        # and the keyword arguments in RANDOMIZATION_SETTINGS (use **).
        # The helper changes these objects in place; do not return a new config.
        raise NotImplementedError("M3: connect the supplied randomization helper here")


def design_tracking(cfg):
    # TODO M1 core: return cfg unchanged. Identify its rewards and failure threshold,
    # then fine-tune the weak policy and compare physical behavior before/after.
    # Optional ablation: change only the action-rate weight; keep termination fixed.
    raise NotImplementedError("M1: implement design_tracking; see the student guide")


def env_config(robot_cfg, args):
    cfg = design_tracking(tracking_env(robot_cfg, args))
    # Supplied: randomize actor observations only; retain the clean critic inputs.
    actor_observations(cfg, noisy=DOMAIN_RANDOMIZATION)
    return cfg
