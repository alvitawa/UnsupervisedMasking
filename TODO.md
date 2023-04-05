
./run.sh python3 main.py --config-name submask_clip_p --multirun main.dataset=sun397,cifar10pgn,flowers10,cifar100,clevr_count,dtd,eurosat,food101,oxfordpets,resisc45,svhn,ucf101 hydra/launcher=joblib hydra.launcher.n_jobs=2 dl.multiprocessing_context=fork

./run.sh main.py --config-name submask_clip --multirun main.dataset=flowers10,cifar100,clevr_count,eurosat,food101,resisc45,svhn,ucf101 hydra/launcher=joblib hydra.launcher.n_jobs=2 dl.multiprocessing_context=fork

./run.sh python3 main.py --config-name baseline_clip --multirun main.dataset=sun397,cifar10pgn,flowers10,cifar100,clevr_count,dtd,eurosat,food101,oxfordpets,resisc45,svhn,ucf101 hydra/launcher=joblib hydra.launcher.n_jobs=2 dl.multiprocessing_context=fork
