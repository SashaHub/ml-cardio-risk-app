import sys
import os
from streamlit.web.cli import main as st_main

def get_path(relative_path):
    """
    Получает абсолютный путь к ресурсу, работает как для режима разработки,
    так и для скомпилированного в .app виде.
    """
    try:
        # PyInstaller создает временную папку _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    # Получаем путь к нашему основному скрипту app.py
    app_path = get_path('app.py')

    # Формируем аргументы командной строки для Streamlit так,
    # как будто мы запускаем его из Python
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        "--server.enableCORS=false"
    ]

    # Вызываем главную функцию Streamlit напрямую
    sys.exit(st_main())