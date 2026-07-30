"""
СЮДА НЕОБХОДИМО ПЕРЕНЕСТИ РЕАЛИЗОВАННЫЕ ФУНКЦИИ ИЗ ЮПИТЕР НОУТБУКА.

+ РЕАЛИЗОВАТЬ функцию preprocess_data, которая должна:
1. Обрабатывать аномальные продажи
2. Заполнять пропуски средним
3. Обогатить датасет признаками, используя функции: 
    create_temporal_features,
    create_avg_sales_feature,
    create_lag_features
    create_rolling_features
"""

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

def create_temporal_features(df):
    pass

def create_avg_sales_feature(df):
    pass

def create_lag_features(df):
    pass

def create_rolling_features(df):
    pass


def preprocess_data(df):
    """
    Полная предобработка данных (идентично логике обучения).
    
    Необходимо обработать аномальные продажи. Пропуски заполняем средним.
    После вызываем функции для вычисления признаков:
        create_temporal_features
        create_avg_sales_feature
        create_lag_features
        create_rolling_features
    """
    raise NotImplementedError('Реализуйте функцию preprocess_data!')
    return df


