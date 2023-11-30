# MNIST Draw Recognition

 MNIST Draw Recognition - приложение для творчества, позволяющее вам создавать уникальные цифры из набора данных MNIST на цифровом холсте. Используйте интуитивный интерфейс для рисования, а встроенная модель машинного обучения мгновенно распознает ваши творения.

## Структура репозитория

- **data:** В этой папке содержатся файлы, необходимые для обучения модели.

- **weights:** Папка с файлами весов обученной модели. Эта информация важна для воспроизведения результатов и использования обученной модели.

- **Main.py:** Основной файл приложения, который создает и запускает интерфейс для рисования и распознавания цифр.

- **MNIST.ipynb:** Jupyter Notebook, в котором содержатся шаги создания, обучения и тестирования нейросети для распознавания цифр из набора данных MNIST. Этот файл полезен для понимания процесса обучения модели.

- **Net.py:** Файл с классом готовой нейросети. Этот класс используется в приложении для распознавания цифр.

## Инструкции по использованию

1. **Установка зависимостей:** Запустите `pip install -r requirements.txt` для установки необходимых библиотек.

2. **Запуск приложения:** Выполните `python Main.py` для запуска приложения. Откроется окно с цифровым холстом, где вы можете рисовать цифры.