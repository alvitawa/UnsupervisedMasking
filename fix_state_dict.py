import sys

import torch

ckpt = sys.argv[1]

checkpoint_path = f'{ckpt}/last.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

drop_keys = []
for param in checkpoint['state_dict'].keys():
    if 'model_shell' in param:
        drop_keys.append(param)

for key in drop_keys:
    checkpoint['state_dict'].pop(key)

par_names = torch.load('parnames.pt')
d = 159

state = checkpoint['optimizer_states'][0]['state']
state = {k - d: v for k, v in state.items() if k >= d}
checkpoint['optimizer_states'][0]['state'] = state

param_group0 = checkpoint['optimizer_states'][0]['param_groups'][0]['params']
param_group0 = [p - d for p in param_group0 if p >= d]
checkpoint['optimizer_states'][0]['param_groups'][0]['params'] = param_group0

param_group1 = checkpoint['optimizer_states'][0]['param_groups'][1]['params']
param_group1 = [p - d for p in param_group1 if p >= d]
checkpoint['optimizer_states'][0]['param_groups'][1]['params'] = param_group1

torch.save(checkpoint, f'{ckpt}/last_fixed.ckpt')

