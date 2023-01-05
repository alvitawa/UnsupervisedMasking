
source load.sh

# set XTRACE environment variable to pass along to the job
export XTRACE=0

srun --nodes=1 --ntasks=1 --partition=defq --time=24:00:00 --gres=gpu:1 "$@"
