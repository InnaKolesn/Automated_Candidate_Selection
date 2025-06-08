import os
import dill
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer
from hdbscan import approximate_predict
from catboost import Pool, CatBoostRanker

from .utils import parse_experience, preprocess_text


class Clusterer:
    """
    Общий кластеризатор: SBERT → PCA.transform → UMAP.transform → HDBSCAN.predict
    Ожидает .dill файл со словарём {'pca': PCA_fitted, 'umap': UMAP_fitted, 'hdbscan': HDBSCAN_fitted}.
    """
    def __init__(
        self,
        dill_path: str,
        sbert_model_name: str = "sberbank-ai/sbert_large_nlu_ru"
    ):
        with open(dill_path, 'rb') as f:
            obj = dill.load(f)
        self._pca = obj['pca']
        self._umap = obj['umap']
        self._hdbscan = obj['hdbscan']
        
        self.sbert_model_name = sbert_model_name
        self._sb_model = None
        
    def _init_sbert(self):
        if self._sb_model is None:
            self._sb_model = SentenceTransformer(self.sbert_model_name)

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._init_sbert()
        clean = [preprocess_text(t) for t in texts]
        return self._sb_model.encode(clean, show_progress_bar=False)

    def predict(self, texts: list[str]) -> np.ndarray:
        emb = self._embed(texts)
        pca_z = self._pca.transform(emb)
        umap_z = self._umap.transform(pca_z).astype(np.float64)

        if self._hdbscan.metric == 'cosine':
            norm = Normalizer('l2')
            umap_z = norm.fit_transform(umap_z)
        labels, _ = approximate_predict(self._hdbscan, umap_z)
        return labels.astype(int)


class StudyDirectionPipeline:
    """
    Pipeline для Study Direction:
      - кластеризация study_direction через свой Clusterer
      - кластеризация selected_direction через отдельный Clusterer
      - Ranker (CatBoost) на дополнительных фичах + selected_cluster
    """
    def __init__(
        self,
        study_cluster_dill: str,
        select_cluster_dill: str,
        ranker_dill: str,
        sbert_model_name: str = "sberbank-ai/sbert_large_nlu_ru",
        cat_features: list[str] = None
    ):
        # кластерер для study_direction
        self.study_clusterer = Clusterer(
            study_cluster_dill,
            sbert_model_name=sbert_model_name
        )
        # кластерер для selected_direction
        self.select_clusterer = Clusterer(
            select_cluster_dill,
            sbert_model_name=sbert_model_name
        )
        # загрузка CatBoostRanker
        with open(ranker_dill, 'rb') as f:
            self._ranker: CatBoostRanker = dill.load(f)

        self.cat_features = cat_features or []

        # if 'selected_cluster' not in self.cat_features:
        #     self.cat_features.append('selected_cluster')

    def cluster_study(self, texts: list[str]) -> np.ndarray:
        return self.study_clusterer.predict(texts)

    def cluster_selected(self, texts: list[str]) -> np.ndarray:
        return self.select_clusterer.predict(texts)

    def score(self,
              selected_texts: list[str],
              additional_features: pd.DataFrame
    ) -> np.ndarray:
        sel_clusters = self.cluster_selected(selected_texts)
        df = additional_features.copy()
        df['selected_cluster'] = sel_clusters
        pool = Pool(data=df, cat_features=self.cat_features)
        return self._ranker.predict(pool)


class ExperienceFieldPipeline:
    """
    Pipeline для Experience Field:
      - кластеризация selected_direction через Clusterer
      - агрегация списков навыков в эмбеддинги
      - Clusterer можно переиспользовать, если нужен
      - Ranker (CatBoost) на дополнительных фичах + selected_cluster
    """
    def __init__(
        self,
        select_cluster_dill: str,
        ranker_dill: str,
        sbert_model_name: str = "sberbank-ai/sbert_large_nlu_ru",
        cat_features: list[str] = None,
        agg: str = 'mean'
    ):
        self.select_clusterer = Clusterer(
            select_cluster_dill,
            sbert_model_name=sbert_model_name
        )
        with open(ranker_dill, 'rb') as f:
            self._ranker: CatBoostRanker = dill.load(f)
        self.cat_features = cat_features or []
        if 'selected_cluster' not in self.cat_features:
            self.cat_features.append('selected_cluster')
        self.agg = agg.lower()
        # SBERT init
        self._sb_model = None
        self.sbert_model_name = sbert_model_name

    def _init_sbert(self):
        if self._sb_model is None:
            self._sb_model = SentenceTransformer(self.sbert_model_name)

    def _aggregate(self, lists_of_skills: list[list[str]]) -> np.ndarray:
        self._init_sbert()
        dim = self._sb_model.get_sentence_embedding_dimension()
        out = []
        for skills in lists_of_skills:
            if not skills:
                out.append(np.zeros(dim))
            else:
                embs = self._sb_model.encode(
                    [self.preprocess(s) for s in skills],
                    show_progress_bar=False
                )
                if self.agg == 'mean':
                    out.append(embs.mean(axis=0))
                else:
                    out.append(embs.max(axis=0))
        return np.vstack(out)

    def cluster_selected(self, texts: list[str]) -> np.ndarray:
        return self.select_clusterer.predict(texts)

    def score(
        self,
        selected_texts: list[str],
        lists_of_skills: list[list[str]],
        additional_features: pd.DataFrame
    ) -> np.ndarray:
        sel_clusters = self.cluster_selected(selected_texts)
        agg_embs = self._aggregate(lists_of_skills)
        df = additional_features.copy()
        df['selected_cluster'] = sel_clusters
        # возможно добавить agg_embs как новые колонки
        pool = Pool(data=df, cat_features=self.cat_features)
        return self._ranker.predict(pool)
