import sys

import torch

ckpt = sys.argv[1]

checkpoint_path = f'data/models/checkpoints/{ckpt}/last.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

print(checkpoint['epoch'])