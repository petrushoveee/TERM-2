from .base_model import BaseModel
import numpy as np
from ultralytics import YOLO
import cv2
import os


class CNNModel(BaseModel):
    """Модель анализа клеток крови на основе предобученной сверточной нейронной сети YOLOv8n."""
    def __init__(self):
        """Инициализация модели CNN, загрузка предобученных весов."""
        # Получаем абсолютный путь к файлу весов weights.pt
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        args_path = os.path.join(base_dir, 'src', 'models', 'weights.pt')
        
        # Загружаем предобученную модель YOLOv8n с использованием указанного файла весов
        # Предполагается, что weights.pt содержит веса модели YOLOv8n, обученной на нужных классах объектов (клетки крови)
        self.model = YOLO(args_path)
        
    def predict(self, image: np.ndarray) -> int:
        """Предсказывает количество клеток на входном изображении с использованием модели CNN.
        
        Args:
            image: Входное изображение в формате numpy array.
            
        Returns:
            int: Количество обнаруженных объектов (клеток) на изображении.
        """
        # Запускаем процесс предсказания (инференса) модели на изображении
        # Изменяем размер изображения до 640x640, так как YOLOv8n часто работает с таким размером
        image = cv2.resize(image, (640, 640))
        results = self.model(image) # Выполняем предсказание
        
        # Считаем общее количество обнаруженных объектов (клеток) во всех результатах предсказания
        count = 0
        for result in results:
            # result.boxes содержит информацию о каждом обнаруженном объекте (граничные рамки, классы, уверенность)
            count += len(result.boxes) # Количество объектов в текущем результате
            
        return count
    
    def train(self, images: list, labels: list) -> None:
        """Метод обучения не реализован, так как используется предобученная модель."""
        pass 