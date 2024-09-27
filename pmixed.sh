#!/bin/bash
#SBATCH --job-name=sampling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=out/extraction_%j.out
#SBATCH --error=out/extraction_%j.err


export HUGGING_FACE_HUB_TOKEN=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

echo "Which python: $(which python)"
echo "Python version: $(python --version)"
echo "Sys.executable: $(python -c 'import sys; print(sys.executable)')"
echo "PATH: $PATH"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"

# Run the evaluation with custom tokenizer
python dpicl-pmixed.py