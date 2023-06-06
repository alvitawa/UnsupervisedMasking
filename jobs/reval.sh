python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset=multicrop_oxfordpets main.train=0 main.load_checkpoint=SUB-1221/last.ckpt &

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset=multicrop_ucf101 main.train=0 main.load_checkpoint=SUB-1219/last.ckpt &

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset=multicrop_dtd main.train=0 main.load_checkpoint=SUB-1216/last.ckpt &

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset=multicrop_flowers main.train=0 main.load_checkpoint=SUB-1220/last.ckpt;

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset=multicrop_eurosat main.train=0 main.load_checkpoint=SUB-1218/last.ckpt &

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar10 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine main.train=0 main.load_checkpoint=SUB-809/last.ckpt dl.num_workers=4  --multirun &

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine main.train=0 main.load_checkpoint=SUB-849/last.ckpt &

python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_sun397 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine main.train=0 main.load_checkpoint=SUB-985/last.ckpt ;