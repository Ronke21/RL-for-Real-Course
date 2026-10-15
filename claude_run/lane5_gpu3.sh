#!/bin/bash
cd "$(dirname "$0")"
B="--reward_shaping custom --kl_type forward --kl_coef 0.1 --max_steps 400"
SHAPE_MODE=norep bash lane.sh 3 phase5 "e5_norep|$B"
SHAPE_MODE=short bash lane.sh 3 phase5 "e5_short|$B"
