import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from scipy.optimize import curve_fit

# Загрузка Excel-файла
file_path = '514.xlsx'  # Путь к файлу Excel
data = pd.read_excel(file_path, header=None)  # Загружаем без заголовков
num = data.shape[1]
x = np.linspace(0, num-1, num-1)

# Выбор определенной строки
row_number = 0  # Номер строки, которую нужно вытащить (например, 10-я строка)
wavelengths = data.iloc[0, 1:]  # Предполагаем, что первый столбец — длины волн
intensities = data.iloc[row_number, 1:].values  # Извлекаем значения из строки (кроме первой ячейки)



def smooth(line, window, order): 
    y_savgol = savgol_filter(line, window_length=window, polyorder=order)
    plt.figure(figsize=(10, 6))
    plt.plot(x, intensities, label="Original Data (Noisy)", alpha=0.5)
    plt.plot(x, y_savgol, label="Savitzky-Golay Smoothing", color='red', linewidth=2)
    plt.xlabel("Pixels")
    plt.ylabel("Intensity")
    plt.title("Savitzky-Golay Smoothing Example")
    plt.legend()
    plt.grid()
    plt.show()
    return y_savgol

line = smooth(intensities, 51, 3)

center = np.argmax(line)
num_pixels = len(line)
pixel_positions = [(i - center) * 0.0155 for i in range(num_pixels)]

    
plt.figure(figsize=(10, 6))
plt.plot(pixel_positions, line, label="Smoothed Line", color='red')
plt.axvline(0, color='blue', linestyle='--', label="Spatial Center")  # Линия центра
plt.xlabel("Pixel Position (Step = 0.0155)")
plt.ylabel("Intensity")
plt.title("Spectral Line with Spatial Center")
plt.legend()
plt.grid()
plt.show()

# Вывод результатов
print(f"Center: {center}")


# Функция для поиска края линии справа
def find_line_end(line, start_index, threshold):
    for i in range(start_index, len(intensities)):
        if line[i] < threshold:
            return i
    return len(line) - 1  # Возвращает последний пиксель, если порог не найден

# Порог "нуля" для интенсивности
threshold = 0.7  # Задайте ваше значение порога

# Поиск края линии справа
line_end_index = find_line_end(line, center, threshold=threshold)

# Генерация 10 равноотдаленных точек между центром и краем
num_points = 10
right_branch_indices = np.linspace(center, line_end_index, num_points, dtype=int)
right_branch_positions = [pixel_positions[i] for i in right_branch_indices]
right_branch_values = [line[i] for i in right_branch_indices]

# Визуализация результата
plt.figure(figsize=(10, 6))
plt.plot(pixel_positions, line, label="Smoothed Line", color='red')
plt.axvline(0, color='blue', linestyle='--', label="Spatial Center")  # Линия центра
plt.scatter(right_branch_positions, right_branch_values, color='green', zorder=5, label="Right Branch Points")
plt.xlabel("Pixel Position (Step = 0.0155)")
plt.ylabel("Intensity")
plt.title("Right Branch of Spectral Line with Threshold")
plt.legend()
plt.grid()
plt.show()

# Вывод результатов
print("Coordinates and values:")
for pos, val in zip(right_branch_positions, right_branch_values):
    print(f"Pos: {pos:.4f}, Value: {val:.4f}")







# # Гауссова функция
# def gaussian(x, A, mu, sigma):
#     return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# # Координаты правой ветки
# x_data = np.array(right_branch_positions)  # Позиции точек
# y_data = np.array(right_branch_values)     # Значения интенсивности

# # Начальные приближения для параметров A, mu, sigma
# initial_guess = [max(y_data), x_data[np.argmax(y_data)], 0.1]

# # Подгонка данных функцией Гаусса
# params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)

# # Полученные параметры
# A_fit, mu_fit, sigma_fit = params
# print(f"Gauss:\A: {A_fit:.4f}\nCenter: {mu_fit:.4f}\nSigma: {sigma_fit:.4f}")

# # Построение исходных точек и аппроксимации
# x_fit = np.linspace(min(x_data), max(x_data), 100)
# y_fit = gaussian(x_fit, *params)

# plt.figure(figsize=(10, 6))
# plt.scatter(x_data, y_data, color='red', label="Data Points")  # Исходные точки
# plt.plot(x_fit, y_fit, color='blue', label="Gaussian Fit")     # Гауссова аппроксимация
# plt.xlabel("Pixel Position")
# plt.ylabel("Intensity")
# plt.title("Gaussian Fit of Right Branch")
# plt.legend()
# plt.grid()
# plt.show()

