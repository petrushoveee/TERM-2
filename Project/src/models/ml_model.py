from .base_model import BaseModel
import numpy as np
import cv2

class MLModel(BaseModel):
    """Модель анализа клеток крови на основе классических методов обработки изображений и поиска контуров."""
    def __init__(self):
        """Инициализация модели ML и определение параметров для обработки изображений."""
        # Параметры для фильтрации и бинаризации изображения
        self.blur_size = (5, 5) # Размер ядра для размытия по Гауссу
        self.threshold_block_size = 11 # Размер блока для адаптивной бинаризации
        self.threshold_C = 2 # Константа для адаптивной бинаризации
        
        # Параметры для фильтрации найденных контуров
        self.min_contour_area = 100 # Минимальная площадь контура, чтобы считать его клеткой
        self.max_contour_area = 1000 # Максимальная площадь контура
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Выполняет предобработку входного изображения для дальнейшего анализа.
        
        Преобразование в оттенки серого, размытие и адаптивная бинаризация.
        
        Args:
            image: Входное изображение в формате numpy array.
            
        Returns:
            np.ndarray: Бинаризованное изображение после предобработки.
        """
        # Преобразуем в оттенки серого, если изображение цветное
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Применяем размытие по Гауссу для уменьшения шума
        image = cv2.GaussianBlur(image, self.blur_size, 0)
        
        # Применяем адаптивную бинаризацию для отделения объектов от фона
        binary = cv2.adaptiveThreshold(
            image,
            255, # Максимальное значение пикселя
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Метод адаптивного порога (по Гауссу)
            cv2.THRESH_BINARY_INV, # Инвертированный бинарный порог (объекты - белые, фон - черный)
            self.threshold_block_size, # Размер окрестности для вычисления порога
            self.threshold_C # Вычитаемая константа из среднего по окрестности
        )
        
        # Морфологические операции для удаления мелкого шума (Open) и закрытия небольших промежутков (Close)
        kernel = np.ones((3, 3), np.uint8) # Ядро для морфологических операций
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # Операция открытия
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # Операция закрытия
        
        return binary
    
    def find_cells(self, binary: np.ndarray) -> list:
        """Находит и фильтрует контуры на бинаризованном изображении для идентификации клеток."""
        # Находим все внешние контуры на бинаризованном изображении
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL, # Извлекаем только внешние контуры
            cv2.CHAIN_APPROX_SIMPLE # Сжимаем горизонтальные, вертикальные и диагональные сегменты до их конечных точек
        )
        
        # Фильтруем найденные контуры по площади, чтобы отобрать только те, которые соответствуют размеру клеток
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour) # Вычисляем площадь контура
            if self.min_contour_area <= area <= self.max_contour_area: # Проверяем, попадает ли площадь в заданный диапазон
                valid_contours.append(contour) # Добавляем подходящий контур в список
                
        return valid_contours
    
    def predict(self, image: np.ndarray) -> int:
        """Предсказывает количество клеток на входном изображении с использованием классических методов.
        
        Выполняет предобработку, поиск контуров и их фильтрацию.
        
        Args:
            image: Входное изображение в формате numpy array.
            
        Returns:
            int: Количество найденных клеток.
        """
        # Выполняем предобработку изображения для подготовки к поиску контуров
        binary = self.preprocess_image(image)
        
        # Находим контуры, соответствующие клеткам
        contours = self.find_cells(binary)
        
        # Возвращаем количество найденных валидных контуров (клеток)
        return len(contours)
    
    def train(self, images: list, labels: list) -> None:
        """Метод обучения не реализован, так как эта модель не требует обучения на данных с метками."""
        pass 