# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 2: Collect expert data for behavior cloning.

Runs a trained RL policy and records (observation, action) pairs.
The environment loop is provided. Students add the data recording code.

Usage:
    python -m course.module_2_behavior_cloning.collect_data \
        --experiment-name m1_frozen --checkpoint course/pretrained/m1/last.ckpt \
        --simulator isaaclab --num-envs 16 --num-steps 1000 --output bc_data.pt
"""

import argparse

parser = argparse.ArgumentParser(description="Collect expert data for behavior cloning")
parser.add_argument("--experiment-name", type=str, required=True, help="Module 1 experiment name (checkpoint in results/<name>/last.ckpt)")
parser.add_argument("--checkpoint", help="Explicit frozen expert checkpoint")
parser.add_argument("--simulator", type=str, required=True, help="Simulator (isaaclab, newton, etc.)")
parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--num-steps", type=int, default=5000, help="Number of steps to collect")
parser.add_argument("--output", type=str, default="bc_data.pt", help="Output file for dataset")
parser.add_argument("--seed", type=int, default=0, help="Environment and collection seed")
args = parser.parse_args()

# Import simulator before torch (required for IsaacLab/IsaacGym)
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch(args.simulator)

import torch  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from course.module_2_behavior_cloning._env_setup import setup_env, setup_agent  # noqa: E402
from course.physical_eval import file_digest  # noqa: E402


def main():
    torch.manual_seed(args.seed)
    checkpoint_path = args.checkpoint or f"results/{args.experiment_name}/last.ckpt"

    print("=" * 50)
    print("MODULE 2: Collecting Expert Data")
    print("=" * 50)
    print(f"  Experiment:  {args.experiment_name}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Simulator:   {args.simulator}")
    print(f"  Num envs:    {args.num_envs}")
    print(f"  Num steps:   {args.num_steps}")
    print(f"  Output:      {args.output}")
    print()

    # Set up environment
    env, resolved_configs, fabric = setup_env(
        checkpoint_path=checkpoint_path,
        simulator=args.simulator,
        num_envs=args.num_envs,
        headless=True,
        disable_terminations=False,
        seed=args.seed,
    )

    # Set up expert agent (needed to get expert actions)
    agent, actor_in_keys = setup_agent(
        checkpoint_path=checkpoint_path,
        env=env,
        resolved_configs=resolved_configs,
        fabric=fabric,
    )

    num_envs = env.simulator.num_envs
    fabric.seed_everything(args.seed)
    device = fabric.device

    # =====================================================================
    # TODO 1: Create data structure to store collected data
    # obs_keys = list(actor_in_keys): these are the inputs available to the actor.
    # Create all_obs (one empty list per key), all_actions and all_episode_ids.
    # Initialize episode_ids with torch.arange(num_envs, device=device).
    # Each environment starts with a unique ID; the supplied reset code advances it.
    # =====================================================================
    obs_keys = list(actor_in_keys)
    all_obs = {k: [] for k in obs_keys}
    all_actions = []
    all_episode_ids = []
    episode_ids = torch.arange(num_envs, device=device)

    # Run the expert policy and collect data
    resets = 0
    with torch.no_grad(), Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Collecting"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("steps"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[info]}"),
    ) as progress:
        # Reset environment and get initial observations
        obs, _ = env.reset()
        task = progress.add_task("collect", total=args.num_steps, info="")

        for step in range(args.num_steps):
            obs_td = TensorDict(obs, batch_size=[num_envs], device=device)

            # Query the expert for the deterministic (mean) action
            model_outs = agent.model(obs_td)
            action = model_outs["mean_action"]

            # =============================================================
            # TODO 2: Record the expert's observation and action
            # Hint: obs is a dict of {key: tensor} from the environment
            # Append detached, cloned CPU snapshots to each storage list.
            # Record only obs_keys and include the current episode_ids.
            # =============================================================
            for k in obs_keys:
                all_obs[k].append(obs[k].detach().cpu().clone())
            all_actions.append(action.detach().cpu().clone())
            all_episode_ids.append(episode_ids.detach().cpu().clone())

            # Step the environment with the expert's action
            obs, rewards, dones, terminated, extras = env.step(action)

            # Reset any environments that finished their episode
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_indices) > 0:
                # A new ID prevents two episodes from being treated as one trajectory.
                episode_ids[done_indices] += num_envs
                obs, _ = env.reset(done_indices)
                resets += len(done_indices)

            progress.update(task, advance=1, info=f"samples={num_envs * (step + 1):,}  resets={resets}")

    # =====================================================================
    # TODO 3: Save the dataset
    # Hint: include args.experiment_name so downstream scripts know
    #        which checkpoint to use for environment setup. Include
    #        expert_checkpoint_sha256 from course.physical_eval.file_digest.
    # Concatenate each list along dim=0. Save obs, actions, episode_ids,
    # experiment_name, expert_checkpoint_sha256 and seed (args.seed).
    # See the README for the expected shapes and the snapshot-copy pitfall.
    # =====================================================================
    dataset = {
        "obs": {k: torch.cat(v, dim=0) for k, v in all_obs.items()},
        "actions": torch.cat(all_actions, dim=0),
        "episode_ids": torch.cat(all_episode_ids, dim=0),
        "experiment_name": args.experiment_name,
        "expert_checkpoint_sha256": file_digest(checkpoint_path),
        "seed": args.seed,
    }
    torch.save(dataset, args.output)
    print(f"Saved {dataset['actions'].shape[0]:,} samples, "
          f"{torch.unique(dataset['episode_ids']).numel()} trajectories to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
