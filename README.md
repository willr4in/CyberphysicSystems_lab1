# Лабораторная работа №1: Семантическая сегментация

## Задание на четвёрку — Исследование моделей семантической сегментации

**Курс:** Киберфизические системы  
**Библиотека:** `segmentation_models_pytorch`

---

## Описание

Исследование моделей семантической сегментации на датасете **Semantic Drone Dataset** (аэрофотоснимки с дронов, 400 изображений, 8 классов).

**Что реализовано:**
1. **Выбор данных и метрик** — Semantic Drone Dataset (Kaggle), метрики: mIoU, Dice, Pixel Accuracy
2. **Бейзлайн** — UNet+ResNet34 (CNN) и UNet+MiT-B0 (Transformer) из `smp`
3. **Улучшенный бейзлайн** — DeepLabV3++ResNet50 и UNet+MiT-B2 с аугментациями, Combined Loss, CosineAnnealing LR
4. **Кастомные модели** — UNet и SegNet, реализованные с нуля (без предобученных весов)
5. **Итоговое сравнение** всех 8 моделей

---

## Требования

- Python 3.9+
- CUDA-совместимый GPU (рекомендуется, но не обязателен — работает и на CPU)
- Аккаунт на [Kaggle](https://www.kaggle.com/) для загрузки датасета (можно и вручную скачать, но размер датасета ~4 ГБ)

---

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone <URL_репозитория>
cd CyberphysicSystems_lab1
```

### 2. Создать виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# или: venv\Scripts\activate  # Windows
```

### 3. Установить зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Настроить Kaggle API (для автоматической загрузки датасета)

1. Зайдите на https://www.kaggle.com/settings
2. В разделе **API** нажмите **Create New Token** — скачается файл `kaggle.json`
3. Поместите его:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
4. Установите права (Linux/macOS):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

> **Альтернатива:** при первом запуске ноутбука `opendatasets` запросит Kaggle username и API key интерактивно.

> **Ручная загрузка:** скачайте датасет с https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset и киньте папку `semantic_drone_dataset/` в корень проекта:
> ```
> semantic_drone_dataset/
> ├── original_images/          ← изображения (.jpg)
> └── label_images_semantic/    ← маски (.png)
> ```

### 5. Запустить Jupyter Notebook

```bash
jupyter notebook lab1_semantic_segmentation.ipynb
```

Или через JupyterLab:
```bash
pip install jupyterlab
jupyter lab
```

### 6. Выполнить ноутбук

Запустите все ячейки последовательно (**Cell → Run All** или `Ctrl+Shift+Enter` по ячейкам).

- По умолчанию: `BASELINE_EPOCHS = 5`, `IMPROVED_EPOCHS = 8`. Если железо позволяет — можно увеличить для лучших результатов

> **Google Colab:** ноутбук запускался в Colab (PyTorch 2.10.0+cu128, GPU T4/A100).

---

## Структура проекта

```
CyberphysicSystems_lab1/
├── README.md                           # Этот файл
├── requirements.txt                    # Зависимости Python
├── lab1_semantic_segmentation.ipynb    # Основной ноутбук с кодом и анализом
└── .gitignore                          # Исключения для git
```

После запуска дополнительно появятся:
```
├── semantic-drone-dataset/             # Загруженный датасет (не в git)
└── best_*.pth                          # Сохранённые веса лучших моделей (не в git)
```

---

## Структура ноутбука

| Раздел | Описание |
|--------|----------|
| 1. Выбор набора данных | Semantic Drone Dataset — обоснование, EDA, распределение классов |
| 2. Метрики качества | mIoU, Dice, Pixel Accuracy — формулы и обоснование |
| 3. Подготовка данных | Dataset class, трансформации, train/val/test split |
| 4. Бейзлайн | UNet+ResNet34 (CNN) и UNet+MiT-B0 (Transformer) |
| 5. Улучшение бейзлайна | Аугментации, DeepLabV3+, UNet+MiT-B2, Combined Loss |
| 6. Кастомные модели | UNet и SegNet с нуля + улучшения |
| 7. Итоговые выводы | Сравнительная таблица всех 8 моделей, анализ |

---

## Датасет

**Semantic Drone Dataset** — аэрофотоснимки городской территории, снятые с дронов.

- **Источник:** https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset
- **Размер:** ~400 изображений (6000×4000 px, ресайз до 256×256)
- **Классы (8):** background, ground, vegetation, building, infrastructure, person, vehicle, water
- **Разбиение:** 70% train / 15% val / 15% test

---

## Модели

| Модель | Тип | Предобучение | Энкодер |
|--------|-----|--------------|---------|
| UNet + ResNet34 | CNN (smp) | ImageNet | ResNet34 |
| UNet + MiT-B0 | Transformer (smp) | ImageNet | MiT-B0 |
| DeepLabV3+ + ResNet50 | CNN (smp) | ImageNet | ResNet50 |
| UNet + MiT-B2 | Transformer (smp) | ImageNet | MiT-B2 |
| Custom UNet | CNN (from scratch) | Нет | Свой |
| Custom SegNet | CNN (from scratch) | Нет | Свой |

---

## Результаты

Итоговое сравнение всех моделей на тестовой выборке (`BASELINE_EPOCHS=5`, `IMPROVED_EPOCHS=8`):

| Модель | mIoU | Dice | Pixel Acc |
|--------|------|------|-----------|
| UNet+ResNet34 (baseline) | 0.3312 | 0.4247 | 0.8006 |
| UNet+MiT-B0 (baseline) | 0.3238 | 0.3966 | 0.8085 |
| DeepLabV3++ResNet50 (improved) | **0.4317** | **0.5241** | **0.8287** |
| UNet+MiT-B2 (improved) | 0.3929 | 0.4781 | 0.8178 |
| Custom UNet (baseline) | 0.2416 | 0.3151 | 0.7389 |
| Custom SegNet (baseline) | 0.1976 | 0.2541 | 0.6881 |
| Custom UNet (improved) | 0.2358 | 0.2969 | 0.7484 |
| Custom SegNet (improved) | 0.1804 | 0.2215 | 0.7140 |

---

## Устранение проблем

| Проблема | Решение |
|----------|---------|
| `CUDA out of memory` | Уменьшите `BATCH_SIZE` (4 или 2) в ячейке конфигурации |
| `opendatasets` не скачивает | Настройте `kaggle.json` или скачайте датасет вручную |
| `mit_b0` не найден | Обновите: `pip install -U segmentation-models-pytorch timm` |
| Медленное обучение на CPU | Уменьшите `BASELINE_EPOCHS` и `IMPROVED_EPOCHS` в ячейке конфигурации |

