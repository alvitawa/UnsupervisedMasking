#python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=oxfordpets dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine model.pret.prune_criterion=topk model.pret.prune_k=0.5 model.pret.kaiming_scores_init=1 main.load_checkpoint=SUB-1381/last.ckpt main.train=0 main.probe=1 &

#python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=eurosat dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine model.pret.prune_criterion=topk model.pret.prune_k=0.5 model.pret.kaiming_scores_init=1 main.load_checkpoint=SUB-1380/last.ckpt main.train=0 main.probe=1 &

#python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=flowers dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine model.pret.prune_criterion=topk model.pret.prune_k=0.5 model.pret.kaiming_scores_init=1 main.load_checkpoint=SUB-1379/last.ckpt main.train=0 main.probe=1 &

#python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=ucf101 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine model.pret.prune_criterion=topk model.pret.prune_k=0.5 model.pret.kaiming_scores_init=1 main.load_checkpoint=SUB-1377/last.ckpt main.train=0 main.probe=1;

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine model.pret.prune_criterion=topk model.pret.prune_k=0.5 model.pret.kaiming_scores_init=1 main.load_checkpoint=SUB-1372/last.ckpt main.train=0 main.probe=1 &

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=sun397 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine model.pret.prune_criterion=topk model.pret.prune_k=0.5 model.pret.kaiming_scores_init=1 main.load_checkpoint=SUB-1368/last.ckpt main.train=0 main.probe=1 ;

#python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=dtd dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine model.pret.prune_criterion=topk model.pret.prune_k=0.5 model.pret.kaiming_scores_init=1 main.load_checkpoint=SUB-1364/last.ckpt main.train=0 main.probe=1 &

#python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=cifar10pgn dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine model.pret.prune_criterion=topk model.pret.prune_k=0.5 model.pret.kaiming_scores_init=1 main.load_checkpoint=SUB-1358/last.ckpt main.train=0 main.probe=1;