pip install spacy pandas numpy
python -m spacy download ru_core_news_sm

import pandas as pd
import json
import os
import spacy
from spacy.training import Example
import random

# Конфигурация
BASE_DATA_PATH = "data/upload"  # Путь к папке с исходными файлами
CSV_PATH = "annotations.csv"    # Путь к CSV с разметкой
MODEL_PATH = "ner_model"        # Куда сохранить модель

# Чтение и подготовка данных
df = pd.read_csv(CSV_PATH)
df['label'] = df['label'].apply(json.loads)

# Группировка по файлам
file_annotations = {}
for _, row in df.iterrows():
    file_path = os.path.join(BASE_DATA_PATH, row['text'].split('/')[-1])
    if file_path not in file_annotations:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        file_annotations[file_path] = {'content': content, 'entities': []}
    
    for ann in row['label']:
        if ann['labels'][0] in ['fio', 'email', 'position']:
            file_annotations[file_path]['entities'].append(
                (ann['start'], ann['end'], ann['labels'][0])
)

# Форматирование данных для spaCy
train_data = []
for file_info in file_annotations.values():
    entities = []
    for start, end, label in file_info['entities']:
        entities.append((start, end, label.upper()))
    train_data.append((file_info['content'], {'entities': entities}))

# Инициализация модели
nlp = spacy.load("ru_core_news_sm")

# Добавление новых labels в NER
ner = nlp.get_pipe("ner")
for label in ["FIO", "EMAIL", "POSITION"]:
    ner.add_label(label)

# Обучение модели
optimizer = nlp.create_optimizer()
for iteration in range(30):
    random.shuffle(train_data)
    losses = {}
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
    print(f"Iteration {iteration}, Losses: {losses}")

# Сохранение модели
nlp.to_disk(MODEL_PATH)
print("Model saved!")

import spacy

# Загрузка модели
nlp = spacy.load("ner_model")

# Тестовый файл
TEST_FILE = "test_file.txt"

def extract_entities(text):
    doc = nlp(text)
    results = {}
    for ent in doc.ents:
        if ent.label_ in ['FIO', 'EMAIL', 'POSITION']:
            results.setdefault(ent.label_, []).append(ent.text)
    return results

if __name__ == "__main__":
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    
    entities = extract_entities(text)
    for entity_type, values in entities.items():
        print(f"{entity_type}: {', '.join(values)}")