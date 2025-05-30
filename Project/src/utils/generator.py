import os
import cv2
import numpy as np

class BloodCellGenerator:
    """Генератор синтетических изображений клеток крови для обучения и тестирования моделей."""
    def __init__(self, data_dir, image_size=(1024, 1024)):
        """Инициализация генератора с путями к данным и целевым размером изображения."""
        self.cells_dir = os.path.join(data_dir, "cells") # Папка с изображениями отдельных клеток
        self.backgrounds_dir = os.path.join(data_dir, "backgrounds") # Папка с фоновыми изображениями
        self.image_size = image_size # Целевой размер генерируемого изображения (ширина, высота)

        self.cells = self.load_cells() # Загружаем изображения клеток
        self.backgrounds = self.load_backgrounds() # Загружаем фоновые изображения
    
    def load_cells(self):
        """Загружает изображения отдельных клеток из указанной папки."""
        cells = []
        for file in os.listdir(self.cells_dir):
            img = cv2.imread(os.path.join(self.cells_dir, file)) # Читаем изображение
            if img is not None:
                cells.append(img) # Добавляем изображение в список, если оно успешно загружено
        return cells

    def load_backgrounds(self):
        """Загружает фоновые изображения из указанной папки и изменяет их размер."""
        backgrounds = []
        for file in os.listdir(self.backgrounds_dir):
            img = cv2.imread(os.path.join(self.backgrounds_dir, file)) # Читаем изображение
            if img is not None:
                img = cv2.resize(img, (500, 500)) # Изменяем размер фонового изображения
                backgrounds.append(img) # Добавляем изображение в список
        return backgrounds

    def generate_background(self):
        """Генерирует сложное фоновое изображение путем смешивания случайных фоновых текстур."""
        # Выбираем случайное фоновое изображение в качестве основы
        canvas = self.backgrounds[np.random.randint(0, len(self.backgrounds))] # Выбираем случайный фон
        canvas = cv2.resize(canvas, self.image_size) # Изменяем размер под целевой

        # Добавляем дополнительные фоновые текстуры с помощью бесшовного клонирования
        for _ in range(10): # Добавляем 10 случайных текстур
            background = self.backgrounds[np.random.randint(0, len(self.backgrounds))] # Выбираем случайную текстуру
            mask = np.full_like(background, 255) # Создаем маску для бесшовного клонирования
            
            h, w = background.shape[:2] # Размеры добавляемой текстуры
            
            # Выбираем случайный центр для размещения текстуры на холсте
            center_y = np.random.randint(0, self.image_size[0])
            center_x = np.random.randint(0, self.image_size[1])

            # Рассчитываем координаты углов добавляемой текстуры относительно центра
            h_left_up = center_y - h // 2
            h_right_down = center_y + h // 2
            w_left_up = center_x - w // 2
            w_right_down = center_x + w // 2

            # Корректируем размер и маску, если текстура выходит за границы холста
            if w_left_up < 0:
                background = background[:, -w_left_up:]
                mask = mask[:, -w_left_up:]
                center_x -= w_left_up // 2

            elif w_right_down > canvas.shape[1]:
                background = background[:, :canvas.shape[1] - w_right_down]
                mask = mask[:, :canvas.shape[1] - w_right_down]
                center_x += (canvas.shape[1] - w_right_down) // 2

            if h_left_up < 0:
                background = background[-h_left_up:, :]
                mask = mask[-h_left_up:, :]
                center_y -= h_left_up // 2

            elif h_right_down > canvas.shape[0]:
                background = background[:canvas.shape[0] - h_right_down, :]
                mask = mask[:canvas.shape[0] - h_right_down, :]
                center_y += (canvas.shape[0] - h_right_down) // 2
            center = (center_x, center_y) # Координаты центра для клонирования

            # Выполняем бесшовное клонирование текстуры на холст
            canvas = cv2.seamlessClone(background, canvas, mask, center, cv2.MIXED_CLONE)

        return canvas
    
    def generate_cells(self, canvas):
        """Размещает случайное количество клеток на фоновом изображении."""
        bboxes = []  # Список для хранения координат ограничивающих рамок клеток
        
        # Генерируем случайное количество клеток для добавления
        for _ in range(np.random.randint(5, 30)): # Количество клеток от 5 до 30
            cell = self.cells[np.random.randint(0, len(self.cells))] # Выбираем случайное изображение клетки
            cell = cv2.resize(cell, (100, 100)) # Изменяем размер клетки
            mask = np.full_like(cell, 255) # Создаем маску для бесшовного клонирования клетки
            
            h, w = cell.shape[:2] # Размеры добавляемой клетки
            
            # Выбираем случайный центр для размещения клетки на холсте
            center_y = np.random.randint(0, self.image_size[0])
            center_x = np.random.randint(0, self.image_size[1])

            # Рассчитываем координаты углов добавляемой клетки относительно центра
            h_left_up = center_y - h // 2
            h_right_down = center_y + h // 2
            w_left_up = center_x - w // 2
            w_right_down = center_x + w // 2

            # Корректируем размер и маску, если клетка выходит за границы холста
            if w_left_up < 0:
                cell = cell[:, -w_left_up:]
                mask = mask[:, -w_left_up:]
                center_x -= w_left_up // 2
                w = cell.shape[1] # Обновляем ширину после обрезки

            elif w_right_down > canvas.shape[1]:
                cell = cell[:, :canvas.shape[1] - w_right_down]
                mask = mask[:, :canvas.shape[1] - w_right_down]
                center_x += (canvas.shape[1] - w_right_down) // 2
                w = cell.shape[1] # Обновляем ширину после обрезки

            if h_left_up < 0:
                cell = cell[-h_left_up:, :]
                mask = mask[-h_left_up:, :]
                center_y -= h_left_up // 2
                h = cell.shape[0] # Обновляем высоту после обрезки

            elif h_right_down > canvas.shape[0]:
                cell = cell[:canvas.shape[0] - h_right_down, :]
                mask = mask[:canvas.shape[0] - h_right_down, :]
                center_y += (canvas.shape[0] - h_right_down) // 2
                h = cell.shape[0] # Обновляем высоту после обрезки

            center = (center_x, center_y) # Координаты центра для клонирования

            # Выполняем бесшовное клонирование клетки на холст
            canvas = cv2.seamlessClone(cell, canvas, mask, center, cv2.MIXED_CLONE)
            
            # Добавляем координаты ограничивающей рамки клетки в список (центр и размеры)
            bboxes.append({
                'center_x': center_x,
                'center_y': center_y,
                'width': w,
                'height': h
            })
                
        return canvas, bboxes
    
    def generate_image(self, return_bboxes=False):
        """Генерирует полное синтетическое изображение с фоном и клетками."""
        background = self.generate_background() # Генерируем фон
        if return_bboxes: # Если требуется вернуть координаты клеток
            image, bboxes = self.generate_cells(background)
            return image, bboxes # Возвращаем изображение и список рамок
        else: # Если координаты клеток не требуются
            image, _ = self.generate_cells(background)
            return image # Возвращаем только изображение 