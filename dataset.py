#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np

def load_tiny_imagenet():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # DINO model requires input images of size 224x224
        transforms.ToTensor(),
    ])

    tiny_imagenet_dataset = datasets.ImageFolder(root='data/tiny-imagenet', transform=transform)
    return tiny_imagenet_dataset

def load_cifar100():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    cifar100_dataset = datasets.CIFAR100(root='data/cifar100', train=True, download=True, transform=transform)
    return cifar100_dataset


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




