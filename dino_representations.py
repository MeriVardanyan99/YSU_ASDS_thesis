#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def generate_dino_representations(dataloader, model):
    model.eval()
    representations = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Calculating DINO representations"):
            features = model(images)
            representations.append(features.numpy())
    
    representations = np.concatenate(representations, axis=0)
    representations = representations.reshape(representations.shape[0], -1)
    return representations


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




