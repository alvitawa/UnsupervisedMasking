#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


# In[ ]:


img = cv2.imread("./img.jpg")

h = min(img.shape[:2])
w = min(img.shape[:2])

center = img.shape
x = center[1]/2 - w/2
y = center[0]/2 - h/2

img = img[int(y):int(y+h), int(x):int(x+w)]
    
img = cv2.resize(img, (512, 512))
img = img[:, :, ::-1]

plt.imshow(img)


# In[3]:


class PixelDataset(Dataset):
    def __init__(self, img):
        self.img = img
        self.h = img.shape[0]
        self.w = img.shape[1]
        
        self._init_pixels(img)
        
    def _init_pixels(self, img):
        # coordinates [[0, 1, 2], [0, 1, 2] [0, 1, 2]] & [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        x_coords, y_coords = torch.meshgrid(torch.arange(self.h), torch.arange(self.w))
        
        z = np.array(img).reshape(self.h*self.w, 3)
        
        self.X = torch.stack((x_coords.flatten(), y_coords.flatten()), -1).float()
        self.y = torch.tensor(z) / 255
        
        print(self.X.shape, self.y.shape)
        print(torch.min(self.y), torch.max(self.y))
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):
        return {
            'X' : self.X[index],
            'y' : self.y[index],
        }


# In[4]:


# simple model: no sinoids

class SimpleModel(torch.nn.Module):
    def __init__(self, in_size=2, hidden_size=128, out_size=3):
        super().__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_size, bias=True),
            torch.nn.Sigmoid(),
        )
        
    def forward(self, X):
        out = self.network(X)
        
        return out
    
# sinoid activation functions
    
class Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return torch.sin(X)
    
class SinoidModel(torch.nn.Module):
    def __init__(self, in_size=2, hidden_size=128, out_size=3):
        super().__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size, bias=True),
            Sin(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            Sin(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            Sin(),
            torch.nn.Linear(hidden_size, out_size, bias=True),
            torch.nn.Sigmoid(),
        )
        
    def forward(self, X):
        out = self.network(X)
        
        return out

# positional encoding
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        
        self.out_size = out_size
        
    def forward(self, X):
        results_sin = torch.stack([torch.sin(2**i * X[:, 0]) for i in range(self.out_size // 4)], -1)
        results_cos = torch.stack([torch.cos(2**i * X[:, 0]) for i in range(self.out_size // 4)], -1)
        
        results2_sin = torch.stack([torch.sin(2**i * X[:, 1]) for i in range(self.out_size // 4)], -1)
        results2_cos = torch.stack([torch.cos(2**i * X[:, 1]) for i in range(self.out_size // 4)], -1)
        
        result = torch.concat((results_sin, results_cos, results2_sin, results2_cos), -1)
        
        return result
    
class PosModel(torch.nn.Module):
    def __init__(self, in_size=2, hidden_size=128, out_size=3):
        super().__init__()
        
        self.network = torch.nn.Sequential(
            PositionalEncoding(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_size, bias=True),
            torch.nn.Sigmoid(),
        )
        
    def forward(self, X):
        out = self.network(X)
        
        return out


# In[5]:


import subnetworks
from subnetworks import submasking

dataset = PixelDataset(img)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = PosModel(hidden_size=128)


weight_summary = ""
for name, param in model.named_parameters():
    row = f"{name}: {param.shape}, {param.numel()} elements, requires_grad={param.requires_grad}\n"
    weight_summary += row
    # logger.experiment[f'global/weight_summary'].log(row)
print(weight_summary)


def par_sel(name, param):
    return 'mask'


model = submasking.SubmaskedModel(model, scale=True, test_input=dataset.X,
                                          parameter_selection=par_sel, k=lambda _e: 0,
                                          prune_criterion='threshold',
                                          scores_init=submasking.normal_scores_init(0.01, 0.0))
print()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch['X'])
        loss = criterion(outputs, batch['y'])
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 0 and i > 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
            running_loss = 0.0
            
    # plot full image prediction
    preds = model(dataset.X)
    preds = preds.reshape((dataset.h, dataset.w, 3))
    
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title(f'epoch {epoch}')
    ax[1].set_title(f'ground truth')
    
    ax[0].imshow(preds.detach().cpu().numpy())
    ax[1].imshow(dataset.img)
    
    if epoch == 99:
        plt.savefig(f"{model.__class__.__name__}.jpg")
    plt.savefig(f'{model.__class__.__name__}_{epoch}.jpg')

print('Finished Training')


# In[ ]:





# In[ ]:





# In[ ]:




