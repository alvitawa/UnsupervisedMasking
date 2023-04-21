NC(5
Cluster 0: 14254 samples
Cluster 1: 8508 samples
Cluster 2: 8731 samples
Cluster 3: 6983 samples
Cluster 4: 11524 samples
Mean: 0.8331199999999999
(50000,) (50000,)
ADJ=0.3424970730040208


#0
./run.sh python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=105 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset_subset_clusters_file=notebook/output_clustersB.pt main.dataset_subset_cluster=0 dl.epochs=526 main.load_checkpoint=SUB-849/last.ckpt
#1
./run.sh python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=176 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset_subset_clusters_file=notebook/output_clustersB.pt main.dataset_subset_cluster=1 dl.epochs=882 main.load_checkpoint=SUB-849/last.ckpt
.
#2
./run.sh python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=172 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset_subset_clusters_file=notebook/output_clustersB.pt main.dataset_subset_cluster=2 dl.epochs=859 main.load_checkpoint=SUB-849/last.ckpt
#3
./run.sh python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=215 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset_subset_clusters_file=notebook/output_clustersB.pt main.dataset_subset_cluster=3 dl.epochs=1074 main.load_checkpoint=SUB-849/last.ckpt
#4
./run.sh python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=130 swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset_subset_clusters_file=notebook/output_clustersB.pt main.dataset_subset_cluster=4 dl.epochs=651 main.load_checkpoint=SUB-849/last.ckpt
