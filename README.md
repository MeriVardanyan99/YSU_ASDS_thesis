# YSU_ASDS_thesis
"Exploratory Analysis of Learned Representations by Self-Supervised Learning in Computer Vision"

STRUCTURE

dataset.py: Contains functions to load and preprocess the datasets.     

dino_representations.py: Generates DINO representations from the datasets.

clustering.py: Handles clustering experiments and clustering quality evaluation.

main.py: Main script to run the project and orchestrate different components.

RESULTS

Clustering results for the Tiny Imagenet and CIFAR-100 datasets are evaluated using Silhouette Score, ARI, and NMI. Spectral Clustering generally outperforms KMeans in terms of ARI and NMI, indicating better cluster consistency and alignment with true clusters. However, Silhouette Scores are relatively low, suggesting that clusters might not be well-separated in the embedding space.
