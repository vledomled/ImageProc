import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd

# Загрузка Excel-файла
file_path = '5932 (2).xlsx'  # Путь к файлу Excel
data = pd.read_excel(file_path, header=None)  # Загружаем без заголовков

# Выбор определенной строки
row_number = 867  # Номер строки, которую нужно вытащить (например, 10-я строка)
wavelengths = data.iloc[0, 1:]  # Предполагаем, что первый столбец — длины волн
intensities = data.iloc[row_number, 1:].values  # Извлекаем значения из строки (кроме первой ячейки)



def smooth(line, window, order):       # Порядок полинома для аппроксимации
    y_savgol = savgol_filter(line, window_length=window, polyorder=order)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Original Data (Noisy)", alpha=0.5)
    plt.plot(x, y_savgol, label="Savitzky-Golay Smoothing", color='red', linewidth=2)
    plt.xlabel("A")
    plt.ylabel("Intensity")
    plt.title("Savitzky-Golay Smoothing Example")
    plt.legend()
    plt.grid()
    plt.show()
    return y_savgol


smooth(intensities, 51, 3)




