import dill
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRanker
from typing import List


class Ranker:
    """
    Обёртка над предобученным CatBoostRanker для ранжирования.
    """
    def __init__(
        self,
        dill_path: str,
        cat_features: List[str]
    ):
        with open(dill_path, 'rb') as f:
            self._model: CatBoostRanker = dill.load(f)
        self.cat_features = cat_features

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        df: DataFrame с категориальными признаками;
        возвращает np.ndarray скорингов.
        """
        N = len(df)
        pool = Pool(
            data=df,
            group_id=np.zeros(N, dtype=int),
            cat_features=self.cat_features
        )
        return self._model.predict(pool)
