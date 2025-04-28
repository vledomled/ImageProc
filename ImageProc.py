import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt

# Параметры для сглаживания
window_length = 51  # Размер окна (нечетное число)
polyorder = 3       # Порядок полинома

# Загрузка данных
file_path = '514.xlsx'  # Укажите путь к вашему файлу
data = pd.read_excel(file_path, header=None)  # Читаем данные без заголовков

num = data.shape
x = np.linspace(0, num[1]-1, num[1]-1)

# Отделяем длины волн (первый столбец) и интенсивности (все остальные)
wavelengths = data.iloc[:, 0].values  # Первый столбец - длины волн
intensities = data.iloc[:, 1:]  # Остальные столбцы - интенсивности
    

# Сглаживание всех строк
smoothed_data = pd.DataFrame()
for i in range(len(intensities)):
    row_data = intensities.iloc[i, :].values
    smoothed_row = savgol_filter(row_data, window_length=window_length, polyorder=polyorder)
    smoothed_data[f"Line_{i+1}"] = smoothed_row

# График сглаженных данных
plt.figure(figsize=(12, 8))
for i in range(len(intensities)):
    plt.plot(x, smoothed_data[f"Line_{i+1}"], label=f"Line {i+1}")
    plt.plot(x , intensities.iloc[i, :], alpha=0.5)
plt.xlabel("Pixels")
plt.ylabel("Smoothed Intensity")
plt.title("Smoothed Spectral Lines")
plt.legend()
plt.grid()
plt.show()



# Определение самой интенсивной линии
total_intensity = smoothed_data.sum(axis=0)
most_intense_line_index = total_intensity.idxmax()
most_intense_line = smoothed_data[most_intense_line_index].values

common_center = np.argmax(most_intense_line)

threshold = float(input("Enter threshold: "))


# Функция для нахождения точки обрезки по порогу
def find_cutoff_point(data, start_index, threshold):
    for i in range(start_index, len(data)):
        if data[i] < threshold:
            return i
    return len(data) - 1  # Если порог не найден, вернём последний индекс



# Определяем точку обрезки для самой интенсивной линии
cutoff_index = find_cutoff_point(most_intense_line, common_center, threshold)


#num_pixels = len(intensities)
pixel_positions = [(i - common_center) * 0.0155 for i in range(num[1]-1)]

# Правая ветка: данные от центра до порога для самой интенсивной линии
num_points = 10
right_branch_indices = np.linspace(common_center, cutoff_index, num_points, dtype=int)
right_branch_positions = [pixel_positions[i] for i in right_branch_indices]
right_branch_values = [most_intense_line[i] for i in right_branch_indices]


results = pd.DataFrame()

for i in range(len(intensities)):
    smoothed_line = smoothed_data[f"Line_{i+1}"].values
    results[i+1] = [smoothed_line[idx] for idx in right_branch_indices]
    
results.loc[-1] = list(wavelengths)  # Добавляем транспонированные данные длин волн
results.index = results.index + 1  # Сдвигаем индексы
results.sort_index(inplace=True)  # Сортируем индексы
    
# output_file = 'right_branch_values.xlsx'
# results.to_excel(output_file, index=False)


plt.figure(figsize=(10, 6))
plt.plot(pixel_positions, most_intense_line, label="Smoothed Line", color='red')
plt.axvline(0, color='blue', linestyle='--', label="Spatial Center")  
plt.scatter(right_branch_positions, right_branch_values, color='green', zorder=5, label="Right Branch Points")
plt.xlabel("Pixel Position (Step = 0.0155)")
plt.ylabel("Intensity")
plt.title("Right Branch of Spectral Line with Threshold")
plt.legend()
plt.grid()
plt.show()


# Гауссова функция
def gaussian_model(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Создание модели
gauss_mod = Model(gaussian_model)


# Длины волн (x значения)
x_values = results.iloc[0, 1:].values  # Первая строка - длины волн

# Результаты аппроксимации
fit_results = []

# Для плавности создаём более плотный массив точек (увеличиваем количество точек)
x_dense = np.linspace(min(x_values), max(x_values), 5000)  # 5000 точек для гладкой аппроксимации

# Апроксимация каждой строки (начиная со второй строки)
for i in range(1, len(results)):
    y_values = results.iloc[i, 1:].values  # Интенсивности текущей строки

    if np.max(y_values) < 1e-3:  # Пропускаем строки с недостаточными данными
        fit_results.append({'Row': i, 'A': None, 'Mu': None, 'Sigma': None})
        print(f"Skipping row {i}: Insufficient data for fitting.")
        continue

    # Начальные приближения
    params = gauss_mod.make_params(A=np.max(y_values), mu=x_values[np.argmax(y_values)], sigma=0.1)

    # Фиттинг модели
    result = gauss_mod.fit(y_values, params, x=x_values)

    if result.success:
        A = result.params['A'].value
        mu = result.params['mu'].value
        sigma = result.params['sigma'].value
        fit_results.append({'Row': i, 'A': A, 'Mu': mu, 'Sigma': sigma})
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, label='Original Data', color='blue')
        plt.plot(x_dense, gaussian_model(x_dense, A, mu, sigma), label='Gaussian Fit (Smooth)', color='red')
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')
        plt.title(f'Gaussian Fit (Row {i})')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        fit_results.append({'Row': i, 'A': None, 'Mu': None, 'Sigma': None})
        print(f"Fit failed for row {i}")

# Сохранение результатов
fit_results_df = pd.DataFrame(fit_results)
output_file = 'gaussian_fit_results_dense.xlsx'
fit_results_df.to_excel(output_file, index=False)




