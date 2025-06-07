import os
import time
import random
import re

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from deep_translator import GoogleTranslator


def contains_latin(text):
    """Проверяет, содержит ли строка латинские символы""" 
    if pd.isna(text):
        return False
    return bool(re.search('[a-zA-Z]', str(text)))

def hybrid_translate(text):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        'client': 'gtx',
        'sl': 'en',
        'tl': 'ru',
        'dt': 't',
        'q': text
    }
    response = requests.get(url, params=params).json()
    return response[0][0][0]  # Получаем перевод как в браузере

def translate_if_latin(text, source='auto', target='ru'):
    """Переводит только если есть латинские символы, логирует перевод"""
    if pd.isna(text):
        return None
    
    text_str = str(text)
    if contains_latin(text_str):
        try:
            # Разделяем строку на части, сохраняя разделители
            parts = re.split('([,;&()])', text_str)

            translated_parts = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Переводим только части с латинскими буквами
                if re.search('[a-zA-Z]', part):
                    try:
                        translated = GoogleTranslator(source=source, target=target).translate(part)
                        # translated = hybrid_translate(part)
                        translated_parts.append(translated)
                        # # Логируем факт перевода
                        # logger.info(f"Переведено: '{part}' -> '{translated}'")
                    except Exception as e:
                        translated_parts.append(part)
                        # logger.error(f"Ошибка перевода '{part}': {str(e)}")
                else:
                    translated_parts.append(part)

            # Собираем обратно в строку
            result = ' '.join(translated_parts)
            return result 
        
        except Exception as e:
            # logger.error(f"Ошибка перевода '{text_str}': {str(e)}")
            return text_str  # В случае ошибки возвращаем оригинал
    else:
        return text_str


def parse_experience(exp_str):
    """
    Преобразует строковое представление опыта работы в число месяцев.
    
    Примеры:
    - '1 год'         -> 12
    - 'год'           -> 12 (интерпретируем как 1 год)
    - 'менее года'    -> 6
    - '7 месяцев'     -> 7
    - '3 4 месяца'    -> (3+4)/2 = 3.5 (можно округлить по необходимости)
    - '1 2 года'      -> ((1+2)/2) * 12 = 18
    """
    
    if pd.isna(exp_str):
        return 0

    exp_str = exp_str.lower().strip()
    
    # Если указано "менее года", возвращаем приблизительно 6 месяцев
    if "менее" in exp_str:
        return 6
    
    # Попытка найти числа в строке
    numbers = re.findall(r'\d+', exp_str)
    
    # Если чисел нет, проверяем наличие ключевых слов
    if not numbers:
        if "год" in exp_str:
            # Если просто написано "год" - считаем это как 1 год
            return 12
        elif "месяц" in exp_str:
            # Если просто написано "месяц" - считаем это как 1 месяц
            return 1
        else:
            return None

    # Преобразуем найденные числа в список целых
    numbers = list(map(int, numbers))
    
    # Если указаны несколько чисел (диапазон), берем среднее
    value = np.mean(numbers) if len(numbers) > 1 else numbers[0]
    
    if "год" in exp_str:
        return value * 12
    elif "месяц" in exp_str:
        return value
    else:
        return value
    
