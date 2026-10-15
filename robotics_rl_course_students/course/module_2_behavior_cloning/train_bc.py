# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 2: Fit a behavior cloning policy (student exercise).

The model, optimizer, validation and checkpoint format are supplied.
Complete two TODOs: split whole trajectories and compute the imitation loss.

Usage:
    python -m course.module_2_behavior_cloning.train_bc \
        --data bc_data.pt --epochs 30 --output bc_policy.pt
"""

import argparse

import torch
import torch.nn as nn
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from tensordict import TensorDict

from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
from protomotions.agents.common.mlp import MLPWithConcat
from course.module_2_behavior_cloning.data_utils import split_trajectories
from course.physical_eval import file_digest


def main():
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy")
    parser.add_argument("--data", type=str, required=True, help="Path to collected dataset (.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="bc_policy.pt", help="Output path for trained model")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or args.lr <= 0:
        parser.error("Epochs, batch size and learning rate must be positive")
    torch.manual_seed(0)

    print("=" * 50)
    print("MODULE 2: Training Behavior Cloning Policy")
    print("=" * 50)

    # Supplied: Load dataset
    print(f"\nLoading dataset from {args.data}...")
    dataset = torch.load(args.data, weights_only=False)
    obs = dataset["obs"]  # dict of {key: tensor[N, dim]}
    actions = dataset["actions"]  # tensor[N, action_dim]
    experiment_name = dataset["experiment_name"]
    obs_keys = list(obs.keys())
    action_dim = actions.shape[-1]

    n = actions.shape[0]
    # TODO 1: call the supplied split_trajectories helper on episode_ids.
    # It returns train_idx and validation_idx, with whole episodes kept together.
    # Do not split individual frames. See data_utils.py for the helper.
    raise NotImplementedError("M2: assign train_idx and validation_idx using split_trajectories")
    print(f"  Samples:    {n}")
    print(f"  Action dim: {action_dim}")
    print(f"  Obs keys:   {obs_keys}")

    # Supplied: Build model using the codebase's config pattern
    config = MLPWithConcatConfig(
        in_keys=obs_keys,
        out_keys=["bc_action"],
        num_out=action_dim,
        layers=[MLPLayerConfig(units=256, activation="relu") for _ in range(3)],
    )
    model = MLPWithConcat(config)

    # Initialize lazy layers with a dummy forward pass
    dummy_td = TensorDict({k: obs[k][:1] for k in obs_keys}, batch_size=[1])
    with torch.no_grad():
        model(dummy_td)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    obs = {k: v.to(device) for k, v in obs.items()}
    actions = actions.to(device)
    train_idx = train_idx.to(device)
    validation_idx = validation_idx.to(device)
    history = []
    with torch.no_grad():
        constant_action = actions[train_idx].mean(dim=0)
        baseline_mse = (actions[validation_idx] - constant_action).square().mean().item()
    print(f"Constant-action validation MSE: {baseline_mse:.6f}")

    # Supplied: Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"\nTraining for {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr})...\n")

    # Supplied: Training loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Training"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("epochs"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[info]}"),
    ) as progress:
        epoch_task = progress.add_task("epochs", total=args.epochs, info="")
        for epoch in range(args.epochs):
            perm = train_idx[torch.randperm(len(train_idx), device=device)]
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(perm), args.batch_size):
                idx = perm[i : i + args.batch_size]

                # Build TensorDict for this batch
                td = TensorDict(
                    {k: obs[k][idx] for k in obs_keys},
                    batch_size=[len(idx)],
                )

                # Forward pass, loss, backprop
                model(td)
                pred = td["bc_action"]
                # TODO 2: assign loss using loss_fn(pred, actions[idx]).
                # pred and the matching expert actions have shape [batch, joints].
                # The model predicts raw policy actions, not joint angles or torques.
                raise NotImplementedError("M2: compute the mean squared imitation loss")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite BC training loss")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(idx)
                num_batches += len(idx)

            avg_loss = epoch_loss / num_batches
            with torch.no_grad():
                validation_sum = 0.0
                for start in range(0, len(validation_idx), args.batch_size):
                    idx = validation_idx[start:start + args.batch_size]
                    td = TensorDict({k: obs[k][idx] for k in obs_keys}, batch_size=[len(idx)])
                    model(td)
                    validation_sum += loss_fn(td["bc_action"], actions[idx]).item() * len(idx)
                validation_loss = validation_sum / len(validation_idx)
            history.append({"train_mse": avg_loss, "validation_mse": validation_loss})
            progress.update(epoch_task, advance=1, info=f"train={avg_loss:.6f} validation={validation_loss:.6f}")

    # Supplied: Save model (config enables reconstruction at test time)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "config": model.config,
        "experiment_name": experiment_name,
        "expert_checkpoint_sha256": dataset["expert_checkpoint_sha256"],
        "dataset_sha256": file_digest(args.data),
        "input_shapes": {k: tuple(v.shape[1:]) for k, v in obs.items()},
        "history": history,
        "constant_action_validation_mse": baseline_mse,
        "training_trajectory_ids": dataset["episode_ids"][train_idx.cpu()].unique(),
        "validation_trajectory_ids": dataset["episode_ids"][validation_idx.cpu()].unique(),
    }, args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
