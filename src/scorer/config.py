# Здесь — зашитые лучшие гиперпараметры, полученные ранее в Optuna:

PIPELINE_PARAMS = {
    'pca_n_components':      0.92,
    'umap_n_neighbors':      10,
    'umap_min_dist':         0.05,
    'umap_n_components':     5,
    'umap_metric':           'cosine',
    'hdbscan_min_cluster_size': 4,
    'hdbscan_min_samples':      2,
    'hdbscan_cluster_epsilon':  0.01,
    'hdbscan_metric':           'euclidean',
}

CB_PARAMS = {
    'iterations':      100,
    'depth':           4,
    'num_leaves':      8,
    'learning_rate':   0.01,
    'l2_leaf_reg':     10.0,
}
