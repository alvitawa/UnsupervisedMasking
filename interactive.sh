
srun --mem=32000M  --gres=gpu:$1 --partition=gpu_shared_course --time=20:00:00 --pty bash -il