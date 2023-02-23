#!/usr/bin/env bash

case "$USER" in
	awarmerdam*)
		module purge

		module load 2022
		module load Anaconda3/2022.05

#		source activate img

		source keys.sh
	;;
	ataboada*)
		module purge

		module add cuda91/toolkit/9.1.85
	;;
	ookah*)
		source keys.sh
	;;
	*)
		source keys.sh
	;;
esac

export HYDRA_FULL_ERROR=1
export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE=1
export TOKENIZERS_PARALLELISM=false
