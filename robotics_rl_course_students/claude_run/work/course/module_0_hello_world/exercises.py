# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repair the frozen walker's live velocity observation pipeline."""
from protomotions.utils import rotations


def world_velocity_to_heading(velocity, root_rotation):
    """Return [forward speed, leftward speed, upward speed] for each robot.

    velocity: [batch, 3] world-frame vectors.
    root_rotation: [batch, 4] unit xyzw quaternions, body-to-world rotation.
    The heading frame follows yaw only, so world-up remains up.
    See course/rotations.html for quaternion direction and helper signatures.
    """
    # TODO: use rotations.calc_heading_quat_inv to obtain world-to-heading
    # rotation, then rotations.quat_rotate to express velocity in that frame.
    # Both helpers use w_last=True. Return shape [batch, 3].
    # Do not invert the full root rotation: that also removes roll and pitch.
    heading_inv = rotations.calc_heading_quat_inv(root_rotation, w_last=True)
    return rotations.quat_rotate(heading_inv, velocity, w_last=True)
