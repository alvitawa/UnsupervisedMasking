#!/usr/bin/env bash

case "$USER" in
	lcur*)
		module purge

		module load 2021
		module load Anaconda3/2021.05

		source activate img

		source keys.sh
	;;
	*)
		source keys.sh
		source venv/bin/activate
	;;
esac

export HYDRA_FULL_ERROR=1
export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE=1
