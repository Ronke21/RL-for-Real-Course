# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Course BeyondMimic tracker adaptation (https://arxiv.org/abs/2508.08241).

Reduced actor state, privileged critic and a raw squared-action-change cost.
This module implements the course tracking recipe, not the diffusion controller.
"""
from examples.experiments.mimic import mlp_bm_l2c2 as bm
from protomotions.agents.common.config import MLPLayerConfig
from protomotions.agents.ppo.config import L2C2Config
from protomotions.envs.component_factories import (
    global_anchor_pos_rew_factory, mimic_target_poses_reduced_coords_factory,
    reduced_coords_obs_factory,
    tracking_error_term_factory,
)
from course.tracking_rewards import action_rate_factory

terrain_config = bm.terrain_config
scene_lib_config = bm.scene_lib_config
motion_lib_config = bm.motion_lib_config


def actor_observations(env, noisy):
    env.observation_components["noisy_reduced_coords_obs"] = reduced_coords_obs_factory(
        use_noisy=noisy, root_height_obs=False, root_vel_obs=False)
    env.observation_components["noisy_mimic_reduced_coords_target_poses"] = mimic_target_poses_reduced_coords_factory(
        use_noisy=noisy, include_dof_vel=True, include_xy_offset=False)


def configure_robot_and_simulator(robot, simulator, args):
    bm.configure_robot_and_simulator(robot, simulator, args)
    robot.reset_noise = None
    simulator.domain_randomization = None


def env_config(robot, args):
    env = bm.env_config(robot, args)
    env.max_episode_length = 300
    for key in ("clean_reduced_coords_obs", "clean_mimic_reduced_coords_target_poses"):
        env.observation_components.pop(key)
    actor_observations(env, noisy=False)
    env.reward_components["action_rate"] = action_rate_factory()
    env.reward_components["global_anchor_pos"] = global_anchor_pos_rew_factory(weight=.5, sigma=.3)
    env.termination_components["tracking_error"] = tracking_error_term_factory(threshold=.5)
    return env


def agent_config(robot, env, args):
    agent = bm.agent_config(robot, env, args)
    agent.l2c2 = L2C2Config(enabled=False)
    agent.model.actor.mu_model.layers = [MLPLayerConfig(units=256, activation="relu") for _ in range(3)]
    agent.model.critic.layers = [MLPLayerConfig(units=256, activation="relu") for _ in range(3)]
    agent.model.in_keys = list(dict.fromkeys(agent.model.actor.in_keys + agent.model.critic.in_keys))
    return agent


def apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg,
                              terrain_cfg, motion_lib_cfg, scene_lib_cfg, args):
    simulator_cfg.domain_randomization = None
    robot_cfg.reset_noise = None
    actor_observations(env_cfg, noisy=False)
    env_cfg.motion_manager.init_start_prob = 1.0
    env_cfg.motion_manager.resample_on_reset = True
