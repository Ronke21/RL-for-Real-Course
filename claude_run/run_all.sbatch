#!/bin/bash
#SBATCH -p p_b200_nlp
#SBATCH -A ug_tsarfaty
#SBATCH -t 03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=toyppo_all
#SBATCH --output=/home/nlp/ronke21/toyppo_link/claude_run/logs/%x_%j.out
#SBATCH --error=/home/nlp/ronke21/toyppo_link/claude_run/logs/%x_%j.err
echo "JobID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
bash "/home/nlp/ronke21/RL for Real Course/toy-ppo/claude_run/run_all.sh"
