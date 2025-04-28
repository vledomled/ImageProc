import pandas as pd
from scipy.ndimage import maximum_position
from scipy.signal import savgol_filter
import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt

# Параметры для сглаживания
window_length = 75  # Размер окна (нечетное число)
polyorder = 3       # Порядок полинома


# Загрузка данных
file_path = input('Enter file name: ') + '.xlsx'  # Укажите путь к вашему файлу
data = pd.read_excel(file_path, header=None)  # Читаем данные без заголовков

name_line = input('Enter line wavelength: ')

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

# Находим 5 самых интенсивных линий
line_sums = smoothed_data.sum(axis=0)
top_5_lines_indices = line_sums.nlargest(5).index

top_5_lines = smoothed_data[top_5_lines_indices]

# Суммируем интенсивности топ-5 линий
sum_top_5_lines = top_5_lines.sum(axis=1)

# Находим максимум и его позицию
max_intensity = sum_top_5_lines.max()

print('Enter max position? (Enter 1 if NO)')
max_position = int(input())
if(max_position == 1):
    max_position = sum_top_5_lines.idxmax()


# Графики 5 линий и линии максимума
plt.figure(figsize=(12, 8))
for line in top_5_lines.columns:
    plt.plot(x, top_5_lines[line], label=f"{line}")

plt.plot(x, sum_top_5_lines, label="Sum of Top 5 Lines", linestyle="--", color="black")
plt.axvline(max_position, color='red', linestyle='--', label=f"Max at {max_position}")
plt.xlabel("Pixels")
plt.ylabel("Intensity")
plt.title("Top 5 Spectral Lines and Maximum Line")
plt.legend()
plt.grid()
plt.show()

# Определение точки обрезки по порогу
threshold = float(input("Enter threshold: "))

def find_cutoff_point(data, start_index, threshold):
    for i in range(start_index, len(data)):
        if data[i] < threshold:
            return i
    return len(data) - 1  # Если порог не найден, вернём последний индекс

# Определяем точку обрезки для суммы топ-5 линий
cutoff_index = find_cutoff_point(sum_top_5_lines, max_position, threshold)

# Правая ветка: данные от центра до порога для суммы топ-5 линий
num_points = 10
pixel_positions = [(i - max_position) * 0.0155 for i in range(len(sum_top_5_lines))]
right_branch_indices = np.linspace(max_position, cutoff_index, num_points, dtype=int)
right_branch_positions = [pixel_positions[i] for i in right_branch_indices]
right_branch_values = [sum_top_5_lines[i] for i in right_branch_indices]

# График правой ветки
plt.figure(figsize=(10, 6))
plt.plot(pixel_positions, sum_top_5_lines, label="Sum of Top 5 Lines", color='blue')
plt.scatter(right_branch_positions, right_branch_values, color='red', zorder=5, label="Right Branch Points")
plt.axvline(0, color='green', linestyle='--', label="Center")
plt.xlabel("Pixel Position (Step = 0.0155)")
plt.ylabel("Intensity")
plt.title("Right Branch of Summed Top 5 Lines")
plt.legend()
plt.grid()
plt.show()


results = pd.DataFrame()



for i in range(len(intensities)):
    smoothed_line = smoothed_data[f"Line_{i+1}"].values
    results[i+1] = [smoothed_line[idx] for idx in right_branch_indices]
    
results.loc[-1] = list(wavelengths)  # Добавляем транспонированные данные длин волн

results.index = results.index + 1  # Сдвигаем индексы
results.sort_index(inplace=True)  # Сортируем индексы

output_file = 'res_bef_gauss.xlsx'
trans_res = results.transpose()
trans_res.to_excel(output_file, index=False)



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
        fit_results.append({'Radius': right_branch_positions[i-1], 'A': None, 'Mu': None, 'Sigma': None})
        print(f"Skipping row {i}: Insufficient data for fitting.")
        continue

    # Начальные приближения
    params = gauss_mod.make_params(A=np.max(y_values), mu=x_values[np.argmax(y_values)], sigma=0.1)

    # Фиттинг модели
    result = gauss_mod.fit(y_values, params, x=x_values)

    if result.success:
        
        mu = result.params['mu'].value
        sigma = result.params['sigma'].value
        A = result.params['A'].value
        Area = result.params['A'].value * sigma * np.sqrt(2 * np.pi)
        fit_results.append({'Radius': right_branch_positions[i-1], name_line : Area})
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, label='Original Data', color='blue')
        plt.plot(x_dense, gaussian_model(x_dense, A, mu, sigma), label='Gaussian Fit', color='red')
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')
        plt.title(f'Gaussian Fit (Row {i})')
        plt.legend()
        plt.grid()
        
        A = result.params['A'].value * sigma * np.sqrt(2 * np.pi)
    else:
        fit_results.append({'Radius': right_branch_positions[i-1], 'A': None, 'Mu': None, 'Sigma': None})
        print(f"Fit failed for row {i}")
        
plt.show()

# Сохранение результатов
fit_results_df = pd.DataFrame(fit_results)
output_file = name_line + '_gauss.xlsx'
fit_results_df.to_excel(output_file, index=False)





