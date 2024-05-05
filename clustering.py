#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

def pca_visualization(representations, labels):
    pca = PCA(n_components=2)
    representations_pca = pca.fit_transform(representations)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        indices = labels == label
        plt.scatter(representations_pca[indices, 0], representations_pca[indices, 1], label=f'Class {label}', alpha=0.7)
    
    plt.title('PCA Visualization of Representations')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_clustering(labels_true, labels_pred, embeddings):
    silhouette = silhouette_score(embeddings, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return silhouette, ari, nmi


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




