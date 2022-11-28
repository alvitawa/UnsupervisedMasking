
source load.sh

srun --mem=32000M  --gres=gpu:1 --partition=gpu --time=24:00:00 "$@"
