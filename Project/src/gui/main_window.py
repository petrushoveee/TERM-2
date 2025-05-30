import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from utils.generator import BloodCellGenerator
from models.ml_model import MLModel
from models.clustering_model import ClusteringModel
from models.cnn_model import CNNModel
from experiment_db import load_results
from preprocessing.filters import ImageFilters

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор клеток крови") # Устанавливаем заголовок окна
        self.current_image = None
        self.setup_ui()
        
        # Инициализация генератора изображений
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.generator = BloodCellGenerator(data_dir, image_size=(640, 640))
        
        # Инициализация моделей анализа
        self.ml_model = MLModel()
        self.clustering_model = ClusteringModel()
        self.cnn_model = CNNModel()
        
    def setup_ui(self):
        # Создание основного фрейма с внутренними отступами
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew") # Растягиваем фрейм на все доступное пространство
        
        # Фрейм для размещения изображения (левая панель)
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10)) # Размещаем слева, добавляем отступ справа
        
        # Область для отображения текущего изображения
        self.image_frame = ttk.LabelFrame(self.left_panel, text="Изображение", padding="10")
        self.image_frame.grid(row=0, column=0, sticky="nsew") # Растягиваем на всю левую панель
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, padx=5, pady=5) # Размещаем метку в центре фрейма изображения
        
        # Фрейм для элементов управления (правая панель)
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky="nsew") # Размещаем справа
        
        # Секция кнопок загрузки и генерации изображений
        self.load_section = ttk.LabelFrame(self.right_panel, text="Загрузка изображения", padding="10")
        self.load_section.grid(row=0, column=0, sticky="ew", pady=(0, 10)) # Растягиваем по ширине, добавляем нижний отступ
        
        # Кнопка для загрузки изображения из файла
        ttk.Button(self.load_section, text="📂 Загрузить изображение", 
                  command=self.load_image).grid(row=0, column=0, padx=5, pady=5, sticky="ew") # Растягиваем по ширине
        # Кнопка для генерации нового синтетического изображения
        ttk.Button(self.load_section, text="🎨 Сгенерировать изображение", 
                  command=self.generate_image).grid(row=1, column=0, padx=5, pady=5, sticky="ew") # Растягиваем по ширине
        
        # Секция выбора и применения фильтров
        self.filter_frame = ttk.LabelFrame(self.right_panel, text="Фильтры обработки", padding="10")
        self.filter_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10)) # Растягиваем по ширине, добавляем нижний отступ
        
        self.filter_var = tk.StringVar() # Переменная для хранения выбранного фильтра
        filters = ["Без фильтра", "Размытие", "Резкость", "Увеличение контраста"]
        self.filter_combo = ttk.Combobox(self.filter_frame, textvariable=self.filter_var, 
                                       values=filters, state="readonly", width=25) # Выпадающий список фильтров
        self.filter_combo.grid(row=0, column=0, padx=5, pady=5)
        self.filter_combo.set(filters[0]) # Устанавливаем значение по умолчанию
        
        # Кнопка для применения выбранного фильтра к текущему изображению
        ttk.Button(self.filter_frame, text="Применить фильтр", 
                  command=self.apply_filter).grid(row=1, column=0, padx=5, pady=5, sticky="ew") # Растягиваем по ширине
        
        # Секция выбора метода анализа
        self.method_frame = ttk.LabelFrame(self.right_panel, text="Метод анализа", padding="10")
        self.method_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10)) # Растягиваем по ширине, добавляем нижний отступ
        
        self.method_var = tk.StringVar() # Переменная для хранения выбранного метода
        methods = ["Классическое ML", "Кластеризация", "Свёрточная сеть"]
        self.method_combo = ttk.Combobox(self.method_frame, textvariable=self.method_var, 
                                       values=methods, state="readonly", width=25) # Выпадающий список методов анализа
        self.method_combo.grid(row=0, column=0, padx=5, pady=5)
        self.method_combo.set(methods[0]) # Устанавливаем значение по умолчанию
        
        # Кнопка для запуска анализа текущего изображения выбранным методом
        ttk.Button(self.method_frame, text="🔍 Анализировать", 
                  command=self.analyze_image).grid(row=1, column=0, padx=5, pady=5, sticky="ew") # Растягиваем по ширине
        
        # Секция отображения результатов анализа
        self.result_frame = ttk.LabelFrame(self.right_panel, text="Результаты анализа", padding="10")
        self.result_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10)) # Растягиваем по ширине, добавляем нижний отступ
        
        self.result_label = ttk.Label(self.result_frame, text="", font=("Arial", 12)) # Метка для отображения результатов
        self.result_label.grid(row=0, column=0, padx=5, pady=5) # Размещаем метку
        
        # Кнопка для открытия окна с таблицей всех экспериментов
        ttk.Button(self.right_panel, text="📊 Таблица экспериментов", 
                  command=self.show_experiments_table).grid(row=4, column=0, padx=5, pady=5, sticky="ew") # Растягиваем по ширине
        
        # Настройка весов для растяжения виджетов при изменении размера окна
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=3) # Левая панель (изображение) получает больший вес
        self.main_frame.columnconfigure(1, weight=1) # Правая панель (управление) получает меньший вес
        self.main_frame.rowconfigure(0, weight=1)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
    def show_experiments_table(self):
        # Загружаем данные всех экспериментов из БД
        df = load_results()
        window = tk.Toplevel(self.root) # Создаем новое окно верхнего уровня
        window.title("Результаты экспериментов") # Устанавливаем заголовок окна
        window.geometry("800x600") # Устанавливаем начальный размер окна
        
        # Создаем фрейм-контейнер с внутренними отступами
        container = ttk.Frame(window)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # Растягиваем контейнер на все окно
        
        # Создаем виджет Treeview для отображения данных в виде таблицы
        tree = ttk.Treeview(container, columns=list(df.columns), show='headings')
        
        # Добавляем заголовки столбцов на основе названий колонок DataFrame
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center') # Устанавливаем ширину и выравнивание столбцов
        
        # Заполняем таблицу данными из DataFrame
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=list(row)) # Вставляем каждую строку как новую запись в таблице
            
        # Добавляем вертикальную и горизонтальную полосы прокрутки к таблице
        scrollbar_y = ttk.Scrollbar(container, orient=tk.VERTICAL, command=tree.yview)
        scrollbar_x = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Размещаем таблицу и полосы прокрутки внутри контейнера
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        # Настраиваем веса для растяжения таблицы и полос прокрутки
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
    
    def display_image(self, image):
        """Отображает изображение в GUI"""
        if image is None:
            return
            
        # Конвертируем изображение из формата BGR (OpenCV) в RGB (PIL)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Изменяем размер изображения для оптимального отображения в метке
        height, width = image.shape[:2]
        max_size = 800 # Максимальный размер по большей стороне
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height)) # Изменяем размер с сохранением пропорций
            
        # Конвертируем изображение в формат, поддерживаемый Tkinter
        image = Image.fromarray(image) # Преобразуем numpy array в объект Image
        photo = ImageTk.PhotoImage(image=image) # Создаем PhotoImage объект
        
        # Обновляем изображение в метке
        self.image_label.configure(image=photo) # Присваиваем новый PhotoImage метке
        self.image_label.image = photo  # Сохраняем ссылку на объект PhotoImage, чтобы избежать сборки мусора
    
    def load_image(self):
        """Открывает диалоговое окно для выбора файла изображения и загружает его."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")] # Фильтр по типам файлов
        )
        if file_path:
            self.current_image = cv2.imread(file_path) # Загружаем изображение с помощью OpenCV
            self.display_image(self.current_image) # Отображаем загруженное изображение
    
    def generate_image(self):
        """Генерирует новое синтетическое изображение клеток крови."""
        self.current_image = self.generator.generate_image() # Генерируем изображение
        self.display_image(self.current_image) # Отображаем сгенерированное изображение
    
    def apply_filter(self):
        """Применяет выбранный фильтр к текущему изображению."""
        if self.current_image is None:
            return # Ничего не делаем, если изображение не загружено
            
        filter_name = self.filter_var.get() # Получаем название выбранного фильтра
        if filter_name == "Без фильтра":
            # Здесь можно добавить логику для сброса фильтров, если нужно вернуться к исходному изображению
            # Но пока просто выходим, так как display_image всегда показывает текущее self.current_image
            return
        
        image = self.current_image # Работаем с текущим изображением
        if filter_name == "Размытие":
            filtered = ImageFilters.blur(image)
        elif filter_name == "Резкость":
            filtered = ImageFilters.sharpen(image)
        elif filter_name == "Увеличение контраста":
            filtered = ImageFilters.contrast(image)
        else:
            filtered = image # Если фильтр не опознан, используем исходное изображение
        self.display_image(filtered) # Отображаем отфильтрованное изображение
        # Важно: Если мы хотим, чтобы фильтры применялись последовательно,
        # нужно сохранять результат фильтрации в self.current_image.
        # Если каждый фильтр должен применяться к исходному изображению,
        # нужно хранить исходное изображение отдельно.
        # Текущая реализация применяет фильтр к последнему отображенному изображению.
        self.current_image = filtered # Обновляем текущее изображение на отфильтрованное
    
    def analyze_image(self):
        """Запускает анализ текущего изображения выбранным методом."""
        if self.current_image is None:
            return # Ничего не делаем, если изображение не загружено
            
        method = self.method_var.get() # Получаем название выбранного метода анализа
        count = 0 # Переменная для хранения результата подсчета клеток
        
        # try:
        # Выбираем и запускаем соответствующую модель
        if method == "Классическое ML":
            count = self.ml_model.predict(self.current_image)
        elif method == "Кластеризация":
            count = self.clustering_model.predict(self.current_image)
        elif method == "Свёрточная сеть":
            count = self.cnn_model.predict(self.current_image)
            
        # Обновляем метку с результатом анализа
        self.result_label.configure(
            text=f"Найдено клеток: {count}", # Форматируем строку результата
            font=("Arial", 12) # Устанавливаем шрифт
        )
