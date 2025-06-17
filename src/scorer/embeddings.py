import threading
from typing import List
import numpy as np 
from sentence_transformers import SentenceTransformer
from .utils import preprocess_text, translate_if_latin

_model_lock = threading.Lock() 
_model_cache = {}

def get_sbert(model_name: str) -> SentenceTransformer: 
    """ Возвращает закешированный экземпляр SBERT. """ 
    with _model_lock: 
        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name) 
        return _model_cache[model_name]

class SBERTEncoder: 
    """ Простая обёртка для получения эмбеддингов списка строк. """ 
    
    def init(self, model_name: str = 'sberbank-ai/sbert_large_nlu_ru'): 
        self.model_name = model_name 
        self._model = None

    def _init(self):
        if self._model is None:
            self._model = get_sbert(self.model_name)

    def encode(self, texts) -> np.ndarray:
        """
        texts: list of preprocessed strings
        """
        self._init()
        texts = texts.apply(translate_if_latin)
        texts = texts.apply(preprocess_text)
        return self._model.encode(texts, show_progress_bar=False)
