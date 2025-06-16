from typing import List
import dill
import numpy as np
import pandas as pd
from .embeddings import SBERTEncoder
from .clusterer  import Clusterer
from .ranker     import Ranker


class StudyScoringPipeline:
    """
    Единый пайплайн:
      1. Кластеризация Selected_Direction
      2. Кластеризация Study_Direction
      3. Ранжирование с помощью CatBoostRanker

    На входе:
      - sel_cluster_dill: путь к .dill кластерера для Selected
      - stu_cluster_dill: путь к .dill кластерера для Study
      - ranker_dill: путь к .dill ранжера
      - sbert_model_name: имя SBERT модели
      - cat_features: список дополнительных категориальных признаков
    """
    def __init__(
        self,
        sel_cluster_dill: str,
        stu_cluster_dill: str,
        ranker_dill: str,
        sbert_model_name: str = 'sberbank-ai/sbert_large_nlu_ru',
        cat_features: List[str] = None
    ):
        # encoder shared
        self.encoder = SBERTEncoder(sbert_model_name)
        # кластереры
        self.sel_cl = Clusterer(sel_cluster_dill, self.encoder)
        self.stu_cl = Clusterer(stu_cluster_dill, self.encoder)
        # ранжер
        with open(ranker_dill, 'rb') as f:
            self.ranker: Ranker = dill.load(f)
        self.cat_features = cat_features or []

    def predict(
        self,
        sel_texts: List[str],
        stu_texts: List[str],
        additional_features: pd.DataFrame
    ) -> np.ndarray:
        # 1) получаем кластеры
        sel_labels = self.sel_cl.predict(sel_texts)
        stu_labels = self.stu_cl.predict(stu_texts)
        # 2) формируем DataFrame для ранжинга
        df = additional_features.copy()
        df['selected_cluster'] = sel_labels
        df['study_cluster']    = stu_labels
        # 3) предсказываем
        return self.ranker.predict(df)
