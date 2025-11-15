from scipy.interpolate import Akima1DInterpolator
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Определение функций

def constrained_gaussian(x, A, sigma, mu_fixed):
    return A * np.exp(-(x - mu_fixed) ** 2 / (2 * sigma ** 2))

def bockasten_abel_transformation(N, coefficients, r_max):
    e = coefficients.T @ N
    return e / (r_max * 0.001)


def fit_and_interpolate(radii, intensities, fine_points=40, mu_fixed=None):
    # сортируем по x
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    intensities = intensities[sorted_indices]

    # если центр не задан – берём по максимуму
    if mu_fixed is None:
        mu_fixed = radii[np.argmax(intensities)]

    try:
        popt, _ = curve_fit(
            lambda x, A, sigma: constrained_gaussian(x, A, sigma, mu_fixed),
            radii,
            intensities,
            p0=[np.max(intensities), (radii.max() - radii.min())/5]
        )
    except RuntimeError:
        logging.warning("Gaussian fit failed.")
        return None, None, None, None

    fine_radii = np.linspace(radii.min(), radii.max(), fine_points)
    smoothed_intensities = constrained_gaussian(fine_radii, *popt, mu_fixed)

    return popt, mu_fixed, fine_radii, smoothed_intensities

def plot_data(x, y, x_fit=None, y_fit=None, title="", xlabel="", ylabel="", labels=("Cu I 510.5 nm")):
    size = {'size' : 14}
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'o', label="Cu I 510.5 nm", color='blue')
    if x_fit is not None and y_fit is not None:
        plt.plot(x_fit, y_fit, color='green')
    plt.xlim(0, 4)
    plt.xlabel(xlabel, fontdict=size)
    plt.ylabel(ylabel, fontdict=size)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def main():

    for j in range(0, 4):
        # Загрузка исходных данных
        name_line = input('Enter line wavelength: ').strip()
        file_path = Path('results') / f'{name_line}_gauss.xlsx'
        data = pd.read_excel(file_path, header=None)

        radius_columns = data.iloc[:, ::2]
        intensity_columns = data.iloc[:, 1::2]

        abel_results = pd.DataFrame()
        gauss_results = pd.DataFrame()

        bockasten_coefficients = np.array([
            [7.625972, 0.463415, 0, 0, 0, 0, 0, 0, 0, 0],
            [-5.800962, 3.606300, 0.323954, 0, 0, 0, 0, 0, 0, 0],
            [-0.584698, -2.951278, 2.653847, 0.263182, 0, 0, 0, 0, 0, 0],
            [-0.339474, -0.182401, -2.058371, 2.198581, 0.227286, 0, 0, 0, 0, 0],
            [-0.197038, -0.214891, -0.138728, -1.666071, 1.918418, 0.202929, 0, 0, 0, 0],
            [-0.126877, -0.134649, -0.162498, -0.112322, -1.434904, 1.723807, 0.185020, 0, 0, 0],
            [-0.088278, -0.092042, -0.105026, -0.133815, -0.095626, -1.278587, 1.578512, 0.171141, 0, 0],
            [-0.064907, -0.066934, -0.073682, -0.087694, -0.115548, -0.084151, -1.164009, 1.464693, 0.159977, 0],
            [-0.048250, -0.049410, -0.053181, -0.060617, -0.074289, -0.100408, -0.072617, -1.070717, 1.381857, 0.251406],
            [-0.044883, -0.045711, -0.048354, -0.053365, -0.061987, -0.076986, -0.104895, -0.086465, -1.037290, 0.984158]
        ])

        for i, column in enumerate(intensity_columns.columns):
            # полный набор: левая + правая ветка
            radii_all = radius_columns.iloc[:, i].dropna().values
            intensities_all = intensity_columns.iloc[:, i].dropna().values

            # убираем дубликаты радиусов
            if len(np.unique(radii_all)) != len(radii_all):
                logging.warning(f"Duplicate radii in Line {i+1} -> keeping first occurrences")
                uniq_r, idx = np.unique(radii_all, return_index=True)
                radii_all = uniq_r
                intensities_all = intensities_all[idx]

            # сортируем по радиусу
            sorted_idx = np.argsort(radii_all)
            radii_all = radii_all[sorted_idx]
            intensities_all = intensities_all[sorted_idx]

            # глобальный гаусс по всем данным, центр фиксируем в 0
            mu_fixed_global = 0.0
            popt_global, mu_fixed_global, fine_r_all, smoothed_all = fit_and_interpolate(
                radii_all, intensities_all, fine_points=80, mu_fixed=mu_fixed_global
            )
            if popt_global is None:
                continue

            # один общий график "сырые точки + глобальный гаусс"
            plot_data(radii_all, intensities_all, fine_r_all, smoothed_all,
                    xlabel='r, mm', ylabel='I, W/m²')

            # значение в центре (r=0) для подстановки, если где-то нет точки
            center_I = constrained_gaussian(0.0, *popt_global, mu_fixed_global)

            # маски веток по знаку
            mask_left = radii_all < 0
            mask_right = radii_all >= 0

            abel_branches = {}

            # --- обработка двух веток с ОДНИМ набором гауссовых параметров ---
            for branch_name, mask in (("left", mask_left), ("right", mask_right)):
                if not np.any(mask):
                    logging.warning(f"No points for {branch_name} branch in Line {i+1}")
                    continue

                # радиусы этой ветки по модулю
                r_branch = np.abs(radii_all[mask])

                # добавляем центр, если его нет
                if not np.any(np.isclose(r_branch, 0.0, atol=1e-9)):
                    r_branch = np.concatenate(([0.0], r_branch))

                # сортируем и убираем дубликаты
                r_branch = np.unique(r_branch)
                r_branch.sort()

                # значения гауссианы для этой ветки (одни и те же параметры для обеих)
                matched_intensities = constrained_gaussian(r_branch, *popt_global, mu_fixed_global)

                # ограничиваем длиной матрицы Боккастена
                n = len(matched_intensities)
                max_n = bockasten_coefficients.shape[0]
                if n > max_n:
                    logging.warning(
                        f"Line {i+1}, {branch_name} branch: {n} points > {max_n}, truncating to first {max_n}"
                    )
                    n = max_n
                    r_branch = r_branch[:n]
                    matched_intensities = matched_intensities[:n]

                r_max = r_branch.max()
                C = bockasten_coefficients[:n, :n]
                abelized_intensities = bockasten_abel_transformation(matched_intensities, C, r_max)

                # сохраняем для общего графика
                abel_branches[branch_name] = (r_branch, abelized_intensities)

                # ---- запись результатов ----
                gauss_block = pd.DataFrame({
                    f'Radius_{branch_name}_{j+1}': pd.Series(r_branch),
                    f'Line_{branch_name}_{j+1}': pd.Series(matched_intensities),
                })
                gauss_results = pd.concat([gauss_results, gauss_block], axis=1)

                abel_block = pd.DataFrame({
                    f'Radius_{branch_name}_{j+1}': pd.Series(r_branch),
                    f'Line_{branch_name}_{j+1}': pd.Series(abelized_intensities),
                })
                abel_results = pd.concat([abel_results, abel_block], axis=1)

            # --- единый график абелизации для двух веток ---
            if abel_branches:
                plt.figure(figsize=(12, 8))
                if "left" in abel_branches:
                    r_left, e_left = abel_branches["left"]
                    plt.plot(-r_left, e_left, label='left branch')   # отражаем в минус
                if "right" in abel_branches:
                    r_right, e_right = abel_branches["right"]
                    plt.plot(r_right, e_right, label='right branch')
                plt.xlabel('Radius (mm)')
                plt.ylabel('Emissivity (Abel)')
                plt.title(f'Abel Transformation (Line {i+1}, both branches, global Gaussian)')
                plt.legend()
                plt.grid()
                plt.show()


    # Выравнивание данных
    data = pd.read_excel('abel_results.xlsx')
    radius_columns = data.iloc[:, ::2]
    intensity_columns = data.iloc[:, 1::2]

    line_index = 0
    radii_line = radius_columns.iloc[:, line_index].dropna().values
    intensities_line = intensity_columns.iloc[:, line_index].dropna().values

    sorted_indices = np.argsort(radii_line)
    radii_line = radii_line[sorted_indices]
    intensities_line = intensities_line[sorted_indices]
    reference_radii = radii_line

    aligned_intensity_data = pd.DataFrame({'Radius': reference_radii})

    for i, column in enumerate(intensity_columns.columns):
        radii = radius_columns.iloc[:, i].dropna().values
        intensities = intensity_columns.iloc[:, i].dropna().values

        if len(np.unique(radii)) != len(radii):
            logging.error(f"Duplicate radii detected in Line {i+1}")
            continue

        sorted_indices = np.argsort(radii)
        radii = radii[sorted_indices]
        intensities = intensities[sorted_indices]

        interpolator = Akima1DInterpolator(radii, intensities)
        interpolated_intensities = interpolator(reference_radii)

        aligned_intensity_data[f'Line_{i+1}'] = interpolated_intensities

    aligned_intensity_data.to_excel('aligned.xlsx', index=False)

    plt.figure(figsize=(12, 8))
    for column in aligned_intensity_data.columns[1:]:
        plt.plot(aligned_intensity_data['Radius'], aligned_intensity_data[column], label=column)
    plt.xlabel('Radius (mm)')
    plt.ylabel('Intensity')
    plt.title('Aligned Intensities for All Lines')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()