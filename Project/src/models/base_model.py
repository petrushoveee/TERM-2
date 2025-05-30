from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Базовый абстрактный класс для моделей анализа клеток крови."""
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """Абстрактный метод для предсказания количества клеток на изображении.
        
        Args:
            image: Входное изображение в формате numpy array (предполагается предобработка).
            
        Returns:
            int: Предсказанное количество обнаруженных клеток.
        """
        pass
    
    @abstractmethod
    def train(self, images: list, labels: list) -> None:
        """Абстрактный метод для обучения модели на предоставленных данных.
        
        Этот метод должен быть реализован только для моделей, которые требуют обучения.
        
        Args:
            images: Список входных изображений.
            labels: Список соответствующих меток (например, количество клеток).
        """
        pass 