
from scipy.interpolate import Akima1DInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model


file_path = '5932.xlsx'  # Замените на путь к вашему файлу
data = pd.read_excel(file_path)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

gauss_model = Model(gaussian)

intensity_columns = data.iloc[:, 1::2]  # Чётные столбцы - интенсивности
radius_columns = data.iloc[:, ::2]  # Нечётные столбцы - радиусы

fit_results = []  # Список для хранения параметров аппроксимации

# Аппроксимация каждой линии
for i, column in enumerate(intensity_columns.columns):
    radii = radius_columns.iloc[:, i].dropna().values  # Радиусы текущей линии
    intensities = intensity_columns.iloc[:, i].dropna().values  # Интенсивности текущей линии

    # Сортировка данных по радиусам
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    intensities = intensities[sorted_indices]

    # Увеличиваем количество точек для аппроксимации в пределах радиусов
    fine_radii = np.linspace(radii.min(), radii.max(), 1000)  # 1000 точек для более плавной аппроксимации

    # Начальные приближения
    initial_params = gauss_model.make_params(A=np.max(intensities), mu=radii[np.argmax(intensities)], sigma=0.5)

    # Фиттинг
    result = gauss_model.fit(intensities, x=radii, params=initial_params)

    # Сохранение параметров
    fit_results.append({
        'Line': column,
        'Amplitude (A)': result.params['A'].value,
        'Mean (mu)': result.params['mu'].value,
        'Sigma': result.params['sigma'].value,
        'Fit Success': result.success
    })

    # Визуализация аппроксимации
    plt.figure(figsize=(10, 6))
    plt.scatter(radii, intensities, label='Original Data', color='blue')
    plt.plot(fine_radii, gaussian(fine_radii, result.params['A'].value, result.params['mu'].value, result.params['sigma'].value),
             label='Gaussian Fit (Smoothed)', color='red')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Intensity')
    plt.title(f'Gaussian Fit for Line {i+1}')
    plt.legend()
    plt.grid()
    plt.show()

# Сохранение параметров аппроксимации
fit_results_df = pd.DataFrame(fit_results)
fit_results_df.to_excel('gaussian_fit_results.xlsx', index=False)


# Загрузка данных
file_path = '5932.xlsx'  # Замените на путь к вашему файлу
data = pd.read_excel(file_path)

# Разделение радиусов и интенсивностей
radius_columns = data.iloc[:, ::2]  # Радиусы в нечётных столбцах
intensity_columns = data.iloc[:, 1::2]  # Интенсивности в чётных столбцах

# Работа с первой линией
line_index = 0  # Первая линия
radii_line = radius_columns.iloc[:, line_index].dropna().values  # Радиусы первой линии
intensities_line = intensity_columns.iloc[:, line_index].dropna().values  # Интенсивности первой линии

# Сортировка данных для первой линии
sorted_indices = np.argsort(radii_line)
radii_line = radii_line[sorted_indices]
intensities_line = intensities_line[sorted_indices]

# Интерполяция для первой линии
interpolator_line = Akima1DInterpolator(radii_line, intensities_line)
interpolated_intensities_line = interpolator_line(radii_line)

# Сохранение радиусов первой линии для использования в других линиях
reference_radii = radii_line

# Создание таблицы для интенсивностей всех линий в радиусах первой линии
aligned_intensity_data = pd.DataFrame({'Radius': reference_radii})

# Проход по каждой линии и интерполяция интенсивностей
for i, column in enumerate(intensity_columns.columns):
    # Радиусы и интенсивности текущей линии
    radii = radius_columns.iloc[:, i].dropna().values
    intensities = intensity_columns.iloc[:, i].dropna().values

    # Сортировка данных
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    intensities = intensities[sorted_indices]

    # Интерполяция для радиусов первой линии
    interpolator = Akima1DInterpolator(radii, intensities)
    interpolated_intensities = interpolator(reference_radii)

    # Добавление данных в таблицу
    aligned_intensity_data[f'Line_{i+1}'] = interpolated_intensities

# Сохранение данных
aligned_intensity_data.to_excel('aligned_intensities_to_line_1.xlsx', index=False)

# Визуализация
plt.figure(figsize=(12, 8))
for column in aligned_intensity_data.columns[1:]:
    plt.plot(aligned_intensity_data['Radius'], aligned_intensity_data[column], label=column)
plt.xlabel('Radius (mm)')
plt.ylabel('Intensity')
plt.title('Aligned Intensities for All Lines Based on Line 1 Radii')
plt.legend()
plt.grid()
#plt.show()

# Визуализация для первой линии (сравнение исходных и интерполированных данных)
plt.figure(figsize=(10, 6))
plt.plot(radii_line, intensities_line, 'o-', label='Original Data (Line 1)')
plt.plot(radii_line, interpolated_intensities_line, 'x-', label='Interpolated Data (Line 1)')
plt.xlabel('Radius (mm)')
plt.ylabel('Intensity')
plt.title('Comparison of Original and Interpolated Data for Line 1')
plt.legend()
plt.grid()
#plt.show()


