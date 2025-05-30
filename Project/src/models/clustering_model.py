from .base_model import BaseModel
import numpy as np
import cv2
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

class ClusteringModel(BaseModel):
    """Модель анализа клеток крови на основе текстурного анализа с использованием фильтров Laws и кластеризации DBSCAN."""
    def __init__(self):
        """Инициализация модели и определение 1D фильтров Laws."""
        # 1D фильтры Laws для выделения различных текстурных признаков
        self.L5 = np.array([1, 4, 6, 4, 1]) # Level (усреднение)
        self.E5 = np.array([-1, -2, 0, 2, 1]) # Edge (край)
        self.S5 = np.array([-1, 0, 2, 0, -1]) # Spot (пятно/точка)
        self.R5 = np.array([1, -4, 6, -4, 1]) # Ripple (волна)
        
        self.filters_1d = [self.L5, self.E5, self.S5, self.R5]
        self.filter_names = ['L5', 'E5', 'S5', 'R5']
        
    def create_laws_filters(self):
        """Создание набора 2D фильтров Laws из 1D фильтров."""
        filters = {}
        for i, f1 in enumerate(self.filters_1d):
            for j, f2 in enumerate(self.filters_1d):
                name = f"{self.filter_names[i]}{self.filter_names[j]}"
                # 2D фильтр получается сверткой (внешним произведением) двух 1D фильтров
                filters[name] = np.outer(f1, f2)
        return filters
    
    def zero_mean(self, image, kernel_size=15):
        """Вычитание локального среднего значения из изображения для удаления фонового шума."""
        # Создаем ядро для усреднения
        mean_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        # Применяем свертку для получения локального среднего
        local_mean = convolve2d(image, mean_kernel, mode='same', boundary='symm')
        # Вычитаем локальное среднее из исходного изображения
        return image - local_mean
    
    def apply_laws_filters(self, image, kernel_size=15):
        """Применяет набор 2D фильтров Laws к изображению и вычисляет текстурную энергию."""
        # Удаляем локальное среднее
        image = self.zero_mean(image, kernel_size=kernel_size)
        # Создаем 2D фильтры Laws
        filters = self.create_laws_filters()
        energy_maps = {}
        # Применяем каждый фильтр и вычисляем энергию
        for name, kernel in filters.items():
            filtered = convolve2d(image, kernel, mode='same', boundary='symm')
            # Вычисляем энергию как среднее абсолютных значений в окне
            energy = cv2.boxFilter(np.abs(filtered), ddepth=-1, ksize=(kernel_size, kernel_size))
            energy_maps[name] = energy
        return energy_maps
    
    def combine_symmetric_energies(self, energy_maps):
        """Объединяет карты энергии от симметричных пар фильтров (например, L5E5 и E5L5)."""
        combined = {
            'E5E5': energy_maps['E5E5'],
            'S5S5': energy_maps['S5S5'],
            'R5R5': energy_maps['R5R5'],
            # Объединяем симметричные пары, усредняя их энергии
            'L5E5': 0.5 * (energy_maps['L5E5'] + energy_maps['E5L5']),
            'L5S5': 0.5 * (energy_maps['L5S5'] + energy_maps['S5L5']),
            'L5R5': 0.5 * (energy_maps['L5R5'] + energy_maps['R5L5']),
            'E5S5': 0.5 * (energy_maps['E5S5'] + energy_maps['S5E5']),
            'E5R5': 0.5 * (energy_maps['E5R5'] + energy_maps['R5E5']),
            'S5R5': 0.5 * (energy_maps['S5R5'] + energy_maps['R5S5']),
        }
        return combined
    
    def cluster_texture(self, image, combined_maps, eps=0.5, min_samples=50):
        """Выполняет кластеризацию текстурных признаков с помощью DBSCAN для идентификации областей клеток."""
        # Собираем текстурные карты и исходное изображение в один стек признаков
        # Изменяем размер карт до 100x100 для уменьшения размерности
        feature_stack = np.dstack([cv2.resize(combined_maps[key], (100, 100)) for key in sorted(combined_maps)])
        # Добавляем измененное изображение как дополнительный признак
        feature_stack = np.dstack([feature_stack, cv2.resize(image, (100, 100))])
        # Преобразуем стек признаков в список векторов признаков для каждого пикселя
        feature_vectors = feature_stack.reshape(-1, 10) # (100*100) x (9 текстур + 1 изображение) = 10000 x 10

        # Применение алгоритма кластеризации DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(feature_vectors) # eps - максимальное расстояние между выборками, min_samples - количество выборок в окрестности для ядра
        labels = clustering.labels_ # Получаем метки кластеров для каждого вектора признаков

        # Подсчитываем количество уникальных кластеров (исключая шумовой кластер с меткой -1)
        cell_count = len(set(labels)) - (1 if -1 in labels else 0)  # Убираем шум
        return cell_count
    
    def predict(self, image: np.ndarray) -> int:
        """Предсказывает количество клеток на входном изображении, применяя последовательность шагов анализа текстур."""
        # Преобразуем в оттенки серого, если изображение цветное
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем размытие для сглаживания изображения перед текстурным анализом
        image = cv2.GaussianBlur(image, (9, 9), 0.5)
        
        # Применяем фильтры Laws и вычисляем карты энергии
        energy_maps = self.apply_laws_filters(image)
        # Объединяем симметричные карты энергии
        combined_maps = self.combine_symmetric_energies(energy_maps)
        
        # Выполняем кластеризацию на основе текстурных признаков и подсчитываем клетки
        return self.cluster_texture(image, combined_maps, eps=10, min_samples=5) # Используем заданные параметры для DBSCAN
    
    def train(self, images: list, labels: list) -> None:
        """Метод обучения не реализован, так как эта модель не требует явного обучения на данных с метками."""
        pass 