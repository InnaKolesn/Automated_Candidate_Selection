import dill
import numpy as np
from sklearn.preprocessing import Normalizer
from hdbscan import approximate_predict
from typing import List
from .embeddings import SBERTEncoder

class Clusterer:
    """
    Загружает предобученную связку PCA+UMAP+HDBSCAN из .dill и выдаёт кластеры по текстам.
    """
    def __init__(self, dill_path: str, encoder: SBERTEncoder):
        # ожидаем словарь {'pca': PCA, 'umap': UMAP, 'hdbscan': HDBSCAN}
        with open(dill_path, 'rb') as f:
            d = dill.load(f)
        self._pca = d['pca']
        self._umap = d['umap']
        self._hdb = d['hdbscan']
        self.encoder = encoder

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        texts → эмбеддинги SBERT → PCA → UMAP → approximate_predict(HDBSCAN) → метки кластеров.
        """
        emb = self.encoder.encode(texts)
        Xp = self._pca.transform(emb)
        Xu = self._umap.transform(Xp).astype(np.float64)
        if getattr(self._hdb, 'metric', '') == 'cosine':
            Xu = Normalizer('l2').transform(Xu)
        labels, _ = approximate_predict(self._hdb, Xu)
        return labels.astype(int)
