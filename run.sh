
source load.sh

srun --mem=32000M  --gres=gpu:1 --partition=gpu_titanrtx --time=24:00:00 "$@"
