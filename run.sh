

case "$USER" in
	awarmerdam*)
    source load.sh

    # set XTRACE environment variable to pass along to the job
    export XTRACE=0

    srun --nodes=1 --ntasks=1 --cpus-per-task=18 --gpus=1 --partition=gpu --time=24:00:00 --cpus-per-task=18 "$@"
	;;
	ook*)
	  docker build . --network=host -t subnetworks
	  docker run -it -e HISTFILE=/opt/app/.bash_history --ipc=host --rm --security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d/ --name subnetworks_run --network=host -v .:/opt/app -v ~/data:/opt/app/data subnetworks bash -c "nvidia-smi; cd /opt/app; ls; source load.sh; $*"
	;;
esac
