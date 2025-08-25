import pandas as pd
from catboost import CatBoostClassifier
from typing import Dict

def predict_scores(model: CatBoostClassifier, raw_df_cleaned: pd.DataFrame, processed_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Получает предсказания от уже загруженной модели.

    Args:
        model (CatBoostClassifier): Загруженный объект модели CatBoost.
        raw_df_cleaned (pd.DataFrame): Очищенный DataFrame со всеми исходными столбцами.
        processed_df (pd.DataFrame): DataFrame только с признаками для модели.
        config (Dict): Словарь с конфигурацией.

    Returns:
        pd.DataFrame: Итоговый DataFrame с добавленными столбцами patient_id и score.
    """
    # Шаг 1: Получение предсказаний (вероятностей класса 1)
    # Модель больше не загружается здесь, а передается в функцию
    scores = model.predict_proba(processed_df)[:, 1]

    # Шаг 2: Создание итогового DataFrame
    result_df = raw_df_cleaned.copy()

    # Шаг 3: Добавление столбца с ID пациента
    # Используем индекс очищенного датафрейма, чтобы он был уникальным
    result_df['patient_id'] = result_df.index
    
    # Шаг 4: Добавление столбца со скорами
    score_col_name = config['app']['output_score_column']
    result_df[score_col_name] = scores

    # Шаг 5: Перемещение новых столбцов в начало для удобства
    all_cols = result_df.columns.tolist()
    new_order = ['patient_id', score_col_name] + [col for col in all_cols if col not in ['patient_id', score_col_name]]
    
    result_df = result_df[new_order]

    return result_df