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


def fit_and_interpolate(radii, intensities, fine_points=40):
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    intensities = intensities[sorted_indices]

    mu_fixed = radii[0]
    try:
        popt, _ = curve_fit(lambda x, A, sigma: constrained_gaussian(x, A, sigma, mu_fixed),
                            radii, intensities, p0=[intensities[0], (radii.max() - radii.min())/5])
    except RuntimeError:
        logging.warning("Gaussian fit failed.")
        return None, None, None

    fine_radii = np.linspace(radii.min(), radii.max(), fine_points)
    smoothed_intensities = constrained_gaussian(fine_radii, *popt, mu_fixed)

    print(fine_radii)
    print(smoothed_intensities)

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
            radii = radius_columns.iloc[:, i].dropna().values
            intensities = intensity_columns.iloc[:, i].dropna().values

            if len(np.unique(radii)) != len(radii):
                logging.warning(f"Duplicate radii in Line {i+1} -> keeping first occurrences")
                uniq_r, idx = np.unique(radii, return_index=True)
                radii = uniq_r
                intensities = intensities[idx]

            popt, mu_fixed, fine_radii, smoothed_intensities = fit_and_interpolate(radii, intensities)
            if popt is None:
                continue


            plot_data(radii, intensities, fine_radii, smoothed_intensities,
                     xlabel='r, mm', ylabel='I, W/m²')

            matched_intensities = constrained_gaussian(radii, *popt, mu_fixed)

            # ---- СКЛЕЙКА РЕЗУЛЬТАТОВ ЧЕРЕЗ CONCAT (длины могут отличаться) ----
            gauss_block = pd.DataFrame({
                f'Radius_{j+1}': pd.Series(radii),
                f'Line_{j+1}': pd.Series(matched_intensities),
            })
            gauss_results = pd.concat([gauss_results, gauss_block], axis=1)

            # Абелевское преобразование: обрезаем матрицу под фактическую длину n
            r_max = radii.max()
            n = len(matched_intensities)
            C = bockasten_coefficients[:n, :n]  # важно, если n != 10
            abelized_intensities = bockasten_abel_transformation(matched_intensities, C, r_max)

            abel_block = pd.DataFrame({
                f'Radius_{j+1}': pd.Series(radii),
                f'Line_{j+1}': pd.Series(abelized_intensities),
            })
            abel_results = pd.concat([abel_results, abel_block], axis=1)

            plot_data(radii, intensities, radii, abelized_intensities,
                    title=f'Abel Transformation for Line {j+1}',
                    xlabel='Radius (mm)', ylabel='Intensity',
                    labels=('Original', 'Abelized'))

        abel_results.to_excel(f'abel_results_{j}.xlsx', index=False)
        gauss_results.to_excel('gauss_results.xlsx', index=False)

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