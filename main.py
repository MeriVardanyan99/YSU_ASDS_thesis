#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import dino_representations
import dataset
import clustering

# Load datasets
tiny_imagenet = dataset.load_tiny_imagenet()
cifar100 = dataset.load_cifar100()

# Create DataLoaders
tiny_imagenet_loader = DataLoader(tiny_imagenet, batch_size=64, shuffle=False)
cifar100_loader = DataLoader(cifar100, batch_size=64, shuffle=False)

# Load DINO model
vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

# Generate representations for Tiny Imagenet and CIFAR-100
tiny_imagenet_representations = dino_representations.generate_dino_representations(tiny_imagenet_loader, vitb16)
cifar100_representations = dino_representations.generate_dino_representations(cifar100_loader, vitb16)

# Save representations
np.save('tiny_imagenet_representations.npy', tiny_imagenet_representations)
np.save('cifar100_representations.npy', cifar100_representations)

# Clustering and visualization
clustering.pca_visualization(tiny_imagenet_representations, tiny_imagenet_loader.dataset.targets)
clustering.pca_visualization(cifar100_representations, cifar100_loader.dataset.targets)

# Clustering evaluation
clustering_algorithms = {
    'KMeans': KMeans(n_clusters=10),
    'SpectralClustering': SpectralClustering(n_clusters=10),
}

for algo_name, algo in clustering_algorithms.items():
    algo.fit(tiny_imagenet_representations)
    labels_pred = algo.labels_
    silhouette, ari, nmi = clustering.evaluate_clustering(tiny_imagenet_loader.dataset.targets, labels_pred, tiny_imagenet_representations)
    print(f"{algo_name} - Tiny Imagenet:")
    print(f"Silhouette Score: {silhouette}, ARI: {ari}, NMI: {nmi}")

for algo_name, algo in clustering_algorithms.items():
    algo.fit(cifar100_representations)
    labels_pred = algo.labels_
    silhouette, ari, nmi = clustering.evaluate_clustering(cifar100_loader.dataset.targets, labels_pred, cifar100_representations)
    print(f"{algo_name} - CIFAR-100:")
    print(f"Silhouette Score: {silhouette}, ARI: {ari}, NMI: {nmi}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




