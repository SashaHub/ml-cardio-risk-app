import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier

# Импортируем наши модули с логикой
from src.preprocessing import preprocess_for_inference
from src.inference import predict_scores

# --- Конфигурация страницы ---
st.set_page_config(
    page_title="Прогнозирование рисков",
    page_icon="⚕️",
    layout="wide"
)

# --- Кэшированные функции для производительности ---

@st.cache_resource
def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Загружает конфигурационный файл."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Ошибка: Конфигурационный файл не найден по пути '{config_path}'")
        return None

@st.cache_resource
def load_model(model_path: str) -> CatBoostClassifier:
    """Загружает модель один раз и кэширует ее."""
    return pd.read_pickle(model_path)

@st.cache_data
def get_shap_values(_model: CatBoostClassifier, data: pd.DataFrame):
    """Рассчитывает SHAP значения один раз и кэширует их для скорости."""
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer(data)
    return shap_values

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Конвертирует DataFrame в CSV формат для скачивания."""
    return df.to_csv(index=False).encode('utf-8')


# --- Загрузка конфигурации и модели в начале ---
config = load_config()
if config is None:
    st.stop()
    
model = load_model(config['paths']['model'])

# --- Боковая панель (Sidebar) ---
st.sidebar.title("ℹ️ О приложении")
st.sidebar.info(
    "Это приложение предназначено для прогнозирования риска неблагоприятных "
    "сердечно-сосудистых событий у пациентов."
    "\n\n**Как использовать:**"
    "\n1. Загрузите файл в формате `.csv` или `.xlsx`."
    "\n2. Дождитесь результатов обработки."
    "\n3. Проанализируйте полученные скоры и скачайте итоговую таблицу."
)


# --- Основной интерфейс приложения ---
st.title(config['app']['title'])
st.markdown(
    "Пожалуйста, загрузите файл с данными пациентов для анализа. "
    "Приложение автоматически обработает данные и добавит столбец со скором риска."
)

uploaded_file = st.file_uploader(
    "Выберите CSV или XLSX файл",
    type=['csv', 'xlsx'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    with st.spinner('Пожалуйста, подождите. Идет анализ данных...'):
        try:
            # --- Шаг 1: Чтение и базовая очистка файла ---
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)

            if '№ п/п' in raw_df.columns:
                raw_df.drop(columns=['№ п/п'], inplace=True)

            for col in raw_df.columns:
                if raw_df[col].dtype == 'object':
                    raw_df[col] = raw_df[col].str.strip()
                    raw_df[col].replace('', np.nan, inplace=True)
            
            raw_df.index.name = 'patient_id'
            
            # --- Шаг 2: Глубокая предобработка данных ---
            processed_df, raw_df_cleaned, all_warnings = preprocess_for_inference(raw_df, config)

            if all_warnings:
                warning_details = "\n\n".join(f"- {w}" for w in all_warnings)
                st.warning(
                    "⚠️ **Внимание: некоторые строки были удалены в процессе очистки данных.**\n\n"
                    f"{warning_details}"
                )
            
            if raw_df_cleaned.empty:
                st.error("❌ После очистки данных не осталось ни одного пациента для анализа. Пожалуйста, проверьте ваш файл.")
                st.stop()
            
            # --- Шаг 3: Получение предсказаний ---
            result_df = predict_scores(model, raw_df_cleaned, processed_df, config)
            st.success('✅ Анализ успешно завершен!')

            # --- Шаг 4: Отображение результатов ---
            st.markdown("### Результаты прогнозирования")
            st.dataframe(result_df)
            st.markdown("---")

            # --- Шаг 5: Визуализация и аналитика ---
            
            # 5.1 Распределение скоров
            st.markdown("### Распределение скоров риска")
            score_col = config['app']['output_score_column']
            fig_hist = px.histogram(
                result_df,
                x=score_col,
                title='Распределение скоров риска среди пациентов',
                labels={score_col: 'Скор риска'},
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            shap_values = get_shap_values(model, processed_df)


            # 5.2 Глобальная важность признаков (Beeswarm plot)
            st.markdown("### Общая важность признаков (Global Feature Importance)")
            st.info(
                "Этот график показывает влияние каждого признака на прогноз для каждого пациента. "
                "Каждая точка — один пациент. Положение по горизонтали показывает, повышает (вправо) или понижает (влево) "
                "значение признака итоговый риск. Цвет точки показывает величину самого признака (красный — высокий, синий — низкий)."
            )
            
            fig_importance, ax_importance = plt.subplots()
            # Заменяем summary_plot на beeswarm, как в оригинальном коде 
            shap.plots.beeswarm(shap_values, max_display=11, show=False)
            st.pyplot(fig_importance)
            plt.close(fig_importance)


            # 5.3 Персональный разбор прогноза (Waterfall plot)
            st.markdown("### Анализ прогноза для отдельного пациента")
            st.info("Выберите ID пациента из таблицы выше, чтобы увидеть, какие факторы и в какой степени (вклад в вероятность) повлияли на его персональный скор риска.")
            
            patient_ids_for_selection = raw_df_cleaned.index.tolist()
            
            selected_patient_id = st.selectbox(
                "Выберите пациента:",
                options=patient_ids_for_selection
            )
            
            if selected_patient_id is not None:
                positional_index = patient_ids_for_selection.index(selected_patient_id)
                
                st.write(f"**Разбор прогноза для пациента {selected_patient_id}**")
                
                fig_waterfall, ax_waterfall = plt.subplots()
                shap.plots.waterfall(shap_values[positional_index], max_display=10, show=False)
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)

            # Кнопка скачивания
            csv_data = convert_df_to_csv(result_df)
            st.download_button(
                label="📥 Скачать результаты в CSV",
                data=csv_data,
                file_name=f'predictions_{uploaded_file.name}.csv',
                mime='text/csv',
            )

        except (ValueError, FileNotFoundError) as e:
            st.error(f"❌ Ошибка обработки: {e}")
        except Exception as e:
            st.error(f"❌ Произошла непредвиденная системная ошибка: {e}")

else:
    st.info("Ожидание загрузки файла...")