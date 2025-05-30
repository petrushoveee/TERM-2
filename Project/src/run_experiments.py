import os
from datetime import datetime
import cv2
from experiment_db import init_db, save_experiment
from models.cnn_model import CNNModel
from models.ml_model import MLModel
from models.clustering_model import ClusteringModel
from utils.generator import BloodCellGenerator

def run_methods_on_image(image_path, ml_model, clustering_model, cnn_model):
    """Загружает изображение по пути и запускает на нем все три модели анализа клеток."""
    image = cv2.imread(image_path) # Загружаем изображение с диска
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}") # Проверяем успешность загрузки

    # Запускаем каждую модель на изображении и получаем результаты
    method1_result = ml_model.predict(image)
    method2_result = clustering_model.predict(image)
    method3_result = cnn_model.predict(image)

    return method1_result, method2_result, method3_result # Возвращаем результаты всех моделей

def process_generated_images(num_images):
    """Генерирует заданное количество синтетических изображений, анализирует их и сохраняет результаты в БД."""
    init_db() # Инициализируем базу данных перед началом экспериментов

    # Определяем базовый путь и путь к данным
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # Инициализируем генератор изображений и модели анализа
    generator = BloodCellGenerator(data_dir)
    ml_model = MLModel()
    clustering_model = ClusteringModel()
    cnn_model = CNNModel()
    
    # Генерируем изображения и проводим эксперименты в цикле
    for i in range(num_images):
        # Генерируем изображение и получаем количество сгенерированных клеток (bboxes)
        image, bboxes = generator.generate_image(return_bboxes=True)
        num_cells = len(bboxes) # Фактическое количество клеток на сгенерированном изображении
        
        # Запускаем модели на сгенерированном изображении
        m1 = ml_model.predict(image)
        m2 = clustering_model.predict(image)
        m3 = cnn_model.predict(image)
        
        # Сохраняем результаты эксперимента в базу данных
        save_experiment(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Текущая дата и время
            real_data_path="", # Путь к реальным данным пустой для сгенерированных изображений
            gen_params=str(num_cells), # Сохраняем фактическое число сгенерированных клеток как параметр генерации
            method1=m1, # Результат ML модели
            method2=m2, # Результат модели кластеризации
            method3=m3 # Результат CNN модели
        )
        print(f"[Сгенерировано {i+1}/{num_images}]: OK") # Выводим прогресс

if __name__ == "__main__":
    """Основная точка входа для запуска массовых экспериментов."""
    # Определяем путь к папке с изображениями для валидации (если нужно обрабатывать реальные данные)
    # В данном случае, запускаем только генерацию изображений
    base_dir = os.path.dirname(os.path.dirname(__file__))
    images_dir = os.path.join(base_dir, "dataset", "val", "images")
    
    # Запускаем процесс обработки сгенерированных изображений (генерируем 50 изображений)
    process_generated_images(num_images=50) 