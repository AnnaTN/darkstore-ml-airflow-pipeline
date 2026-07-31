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
    df = df.copy()
    
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    return df

def create_avg_sales_feature(df):
    df = df.copy()
    
    df = df.sort_values(["store", "dept", "date"])
    
    df['avg_sales_before'] = df.groupby(['store', 'dept'])['weekly_sales'].transform(lambda x: x.shift().expanding().mean())

    return df

def create_lag_features(df):
    df = df.copy()

    df = df.sort_values(["store", "dept", "date"])

    for i in [1, 2, 4]:
        df[f'sales_{i}week_ago'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(i)

    return df

def create_rolling_features(df):
    df = df.copy()

    df = df.sort_values(["store", "dept", "date"])

    df['mean_sales_2week'] = df.groupby(['store', 'dept'])['weekly_sales'].transform(lambda x: x.shift(1).rolling(2).mean())
    df['mean_sales_4week'] = df.groupby(['store', 'dept'])['weekly_sales'].transform(lambda x: x.shift(1).rolling(4).mean())

    return df


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
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])

    for i in range(1, 6):
        df[f'factor{i}'] = df[f'factor{i}'].fillna(df[f'factor{i}'].mean())

    df.loc[df['weekly_sales'] < 0, 'weekly_sales'] = 0

    df = create_temporal_features(df)
    df = create_avg_sales_feature(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    
    return df