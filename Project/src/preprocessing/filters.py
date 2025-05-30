import cv2
import numpy as np

class ImageFilters:
    @staticmethod
    def blur(image):
        """Применяет фильтр Гаусса для размытия изображения."""
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    @staticmethod
    def sharpen(image):
        """Применяет ядро свертки для увеличения резкости изображения."""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def gradient(image):
        """Вычисляет градиент изображения по осям X и Y с использованием оператора Собеля и возвращает величину градиента."""
        # Применяем оператор Собеля для нахождения градиентов по X и Y
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        # Вычисляем величину градиента как корень из суммы квадратов градиентов
        return np.sqrt(sobelx**2 + sobely**2)
    
    @staticmethod
    def contrast(image, alpha=1.5, beta=0):
        """Регулирует контраст и яркость изображения с помощью линейного преобразования."""
        # Применяем преобразование new_image = alpha*old_image + beta
        # alpha: коэффициент усиления (влияет на контраст, >1 увеличивает, <1 уменьшает)
        # beta: сдвиг (влияет на яркость, >0 увеличивает)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted