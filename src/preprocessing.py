import pandas as pd
import re
from typing import Dict, List, Tuple

def _get_mandatory_source_columns(config: Dict) -> List[str]:
    """
    Определяет список обязательных столбцов с русскими названиями на основе
    финальных признаков, необходимых для модели.
    """
    features_to_use = config['features']['features_to_use']
    column_mapping = config['preprocessing']['column_mapping']
    
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    
    mandatory_russian_cols = []
    
    for feature in features_to_use:
        if feature in reverse_mapping:
            mandatory_russian_cols.append(reverse_mapping[feature])
        elif feature == 'Dose of Statins':
            if 'Statins' in reverse_mapping:
                 mandatory_russian_cols.append(reverse_mapping['Statins'])
            else:
                 raise ValueError("Конфигурация неполная: отсутствует маппинг для 'Statins'")

    return list(set(mandatory_russian_cols))

def preprocess_for_inference(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Выполняет предобработку данных. Приводит типы к числовым, удаляет строки
    с неконвертируемыми значениями или пропусками и возвращает предупреждения.
    """
    df_processed = df.copy()
    all_warnings = []

    # Шаг 0: Очистка названий столбцов
    cleaned_columns = [re.sub(r'\s+', ' ', col).strip() for col in df_processed.columns]
    df_processed.columns = cleaned_columns

    # Шаг 1: Проверка наличия обязательных исходных столбцов
    mandatory_russian_cols = _get_mandatory_source_columns(config)
    missing_mandatory_cols = [col for col in mandatory_russian_cols if col not in df_processed.columns]
    if missing_mandatory_cols:
        raise ValueError(
            f"Ошибка: В загруженном файле отсутствуют обязательные столбцы: {', '.join(missing_mandatory_cols)}"
        )
    
    # Шаг 2: Переименование столбцов
    df_processed.rename(columns=config['preprocessing']['column_mapping'], inplace=True)

    # Шаг 3: Создание новых признаков
    if 'Statins' in df_processed.columns:
        df_processed['Dose of Statins'] = (df_processed['Statins'] == 2).astype(int)

    # ↓↓↓ НОВАЯ ЛОГИКА ПРЕОБРАЗОВАНИЯ ТИПОВ И УДАЛЕНИЯ ↓↓↓
    features_to_use = config['features']['features_to_use']
    type_error_indices = set()

    # Шаг 4: Принудительное преобразование к числовому типу
    for feature in features_to_use:
        if feature in df_processed.columns:
            original_col = df_processed[feature].copy()
            # errors='coerce' превратит все, что не является числом, в NaN (пустое значение)
            df_processed[feature] = pd.to_numeric(df_processed[feature], errors='coerce')
            
            # Находим строки, где преобразование не удалось
            error_mask = df_processed[feature].isnull() & original_col.notnull()
            
            if error_mask.any():
                problem_indices = df_processed[error_mask].index
                for idx in problem_indices:
                    bad_value = original_col.loc[idx]
                    warning = (
                        f"**Пациент (индекс {idx}):** в столбце `{feature}` найдено нечисловое значение "
                        f"**'{bad_value}'**. Строка будет удалена."
                    )
                    all_warnings.append(warning)
                type_error_indices.update(problem_indices)

    # Шаг 5: Удаление строк с любыми пропусками (изначальными или после шага 4)
    nan_mask = df_processed[features_to_use].isnull().any(axis=1)
    all_dropped_indices = df_processed[nan_mask].index
    
    # Создаем предупреждение для строк, где были пропуски изначально
    original_nan_indices = set(all_dropped_indices) - type_error_indices
    if original_nan_indices:
        indices_str = ', '.join(map(str, sorted(list(original_nan_indices))))
        warning = (
            f"**Пациенты (индексы {indices_str}):** строки удалены из-за изначально "
            "пропущенных значений в ключевых столбцах."
        )
        all_warnings.append(warning)
    
    # Окончательно чистим датафрейм
    df_cleaned = df_processed.drop(index=all_dropped_indices)

    return df_cleaned[features_to_use], df_cleaned, all_warnings