import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from lmfit import Model
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(filepath: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(filepath, header=None)
    except Exception as e:
        logging.error(f"Failed to load file {filepath}: {e}")
        raise

def cut_line(data: pd.DataFrame, bottom: float, top: float) -> pd.DataFrame:
    return data[(data.iloc[:, 0] >= bottom) & (data.iloc[:, 0] <= top)]

def smooth_data(intensities: pd.DataFrame, window_length: int, polyorder: int) -> pd.DataFrame:
    smoothed = pd.DataFrame()
    for idx in range(len(intensities)):
        row = intensities.iloc[idx].values
        if len(row) < window_length:
            raise ValueError(f"Row {idx} too short for smoothing window.")
        smoothed_row = savgol_filter(row, window_length=window_length, polyorder=polyorder)
        smoothed[f"Line_{idx+1}"] = smoothed_row
    return smoothed

def find_top_lines(smoothed: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    sums = smoothed.sum(axis=0)
    top_indices = sums.nlargest(top_n).index
    return smoothed[top_indices]

def find_cutoff_right(data, start_index: int, threshold: float) -> int:
    """Граница справа (первый индекс, где сигнал падает ниже threshold)."""
    for i in range(start_index, len(data)):
        if data[i] < threshold:
            return i
    return len(data) - 1

def find_cutoff_left(data, start_index: int, threshold: float) -> int:
    """Граница слева (первый индекс, где сигнал падает ниже threshold)."""
    for i in range(start_index, -1, -1):
        if data[i] < threshold:
            return i
    return 0

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]

    if x_values.size < 4:
        logging.warning(f"Not enough points for fit: {x_values.size}")
        return None

    gauss_mod = Model(gaussian)
    params = gauss_mod.make_params(
        A=np.max(y_values),
        mu=x_values[np.argmax(y_values)],
        sigma=(x_values.max() - x_values.min()) / 6 or 0.1  # разумная начальная оценка
    )
    try:
        result = gauss_mod.fit(y_values, params, x=x_values)
        if result.success:
            return {
                'A': result.params['A'].value,
                'Mu': result.params['mu'].value,
                'Sigma': result.params['sigma'].value
            }
        else:
            logging.warning("Fit unsuccessful.")
            return None
    except Exception as e:
        logging.error(f"Fit error: {e}")
        return None

def main():
    file_base = input('Enter file name (without .xlsx): ').strip()
    name_line = input('Enter line wavelength: ').strip()
    window_length =  75
    polyorder = 3

    bottom_wl = float(input('Enter bottom wavelength: '))
    top_wl = float(input('Enter top wavelength: '))

    file_path = Path(file_base + '.xlsx')
    data = load_data(file_path)

    data = cut_line(data, bottom_wl, top_wl)
    print(f"Cutted file length {len(data)}")

    wavelengths = data.iloc[:, 0].values
    intensities = data.iloc[:, 1:]
    x_pixels = np.linspace(0, intensities.shape[1] - 1, intensities.shape[1])

    smoothed = smooth_data(intensities, window_length, polyorder)
    top_lines = find_top_lines(smoothed)
    sum_top = top_lines.sum(axis=1)  # сумма по 5 линиям → зависимость от пикселя

    plt.figure(figsize=(6, 8))
    for line in top_lines.columns:
        plt.plot(x_pixels, top_lines[line], label=f"{line}")
    plt.plot(x_pixels, sum_top, label="Sum of Top 5 Lines", linestyle="--")
    plt.xlabel("Pixels")
    plt.ylabel("Intensity")
    plt.title("Top 5 Spectral Lines and Summed Intensity")
    plt.legend()
    plt.grid()
    plt.show()

    # Центр (по пикселям)
    sum_top_arr = sum_top.values
    max_pos_auto = int(np.argmax(sum_top_arr))
    print('Max pos (auto): ', max_pos_auto)

    threshold = float(input('Enter threshold: '))

    # если порог >= пику, слегка уменьшим его, чтобы вообще были ветки
    if threshold >= sum_top_arr[max_pos_auto]:
        logging.warning(
            "Threshold is >= peak value, adjusting to 0.5 * peak to avoid degenerate selection."
        )
        threshold = 0.5 * sum_top_arr[max_pos_auto]

    manual = input('Enter max position manually (press ENTER to use auto): ').strip()
    if manual:
        max_pos = int(manual)
    else:
        max_pos = max_pos_auto

    # Границы слева/справа
    left_cutoff_idx = find_cutoff_left(sum_top_arr, max_pos, threshold)
    right_cutoff_idx = find_cutoff_right(sum_top_arr, max_pos, threshold)

    # Координаты относительно центра
    pixel_positions = [(i - max_pos) * 0.0155 for i in range(len(sum_top_arr))]

    # по 9 точек слева и справа + центр
    num_points_side = 9
    left_indices = np.linspace(left_cutoff_idx, max_pos, num_points_side + 1, dtype=int)
    right_indices = np.linspace(max_pos, right_cutoff_idx, num_points_side + 1, dtype=int)

    selected_indices = np.unique(np.concatenate([left_indices, right_indices]))
    selected_positions = [pixel_positions[i] for i in selected_indices]

    if len(selected_indices) < 3:
        logging.error(
            f"Too few unique points selected ({len(selected_indices)}). "
            "Try lowering the threshold."
        )
        return

    # График обеих веток с отмеченными точками
    plt.figure(figsize=(10, 6))
    plt.plot(pixel_positions, sum_top_arr, label="Sum of Top 5 Lines")
    plt.scatter(
        [pixel_positions[i] for i in selected_indices],
        [sum_top_arr[i] for i in selected_indices],
        zorder=5,
        label="Selected Points"
    )
    plt.axvline(0, linestyle='--', label="Center")
    plt.xlabel("Pixel Position (Step = 0.0155)")
    plt.ylabel("Intensity")
    plt.title("Both Branches of Summed Top 5 Lines")
    plt.legend()
    plt.grid()
    plt.show()

    results = pd.DataFrame()
    for idx in range(len(intensities)):  # по всем строкам (длина волны)
        smoothed_line = smoothed[f"Line_{idx+1}"].values  # зависимость от пикселя
        results[idx+1] = [smoothed_line[i] for i in selected_indices]

    results.loc[-1] = list(wavelengths)
    results.index = results.index + 1
    results.sort_index(inplace=True)  # теперь 0-я строка — длина волны

    (Path.cwd() / 'results').mkdir(exist_ok=True)
    header = selected_positions.copy()
    header.insert(0, 'Wavelength')
    results.transpose().to_excel(Path('results') / f'{name_line}_bef_gauss.xlsx',
                                 index=False, header=header)

    fit_results = []

    x_vals = results.iloc[0, 1:].values
    x_dense = np.linspace(np.min(x_vals), np.max(x_vals), 5000)

    for i in range(1, len(results)):
        y_vals = results.iloc[i, 1:].values

        if np.max(y_vals) < 1e-3:
            logging.info(f"Skipping row {i}: low signal.")
            continue

        fit = fit_gaussian(x_vals, y_vals)
        if fit:
            area = fit['A'] * fit['Sigma'] * np.sqrt(2 * np.pi)

            radius_idx = i - 1
            radius = selected_positions[radius_idx]

            fit_results.append({'Radius': radius, name_line: area})

            plt.figure(figsize=(10, 6))
            plt.scatter(x_vals, y_vals, label='Data')
            plt.plot(x_dense, gaussian(x_dense, fit['A'], fit['Mu'], fit['Sigma']),
                     label='Fit')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.title(f'Gaussian Fit - Radius {radius:.4f}')
            plt.legend()
            plt.grid()
        else:
            logging.warning(f"Fit failed for row {i}.")

    plt.show()

    fit_df = pd.DataFrame(fit_results)
    fit_df.to_excel(Path('results') / f'{name_line}_gauss.xlsx',
                    index=False, header=False)

if __name__ == "__main__":
    main()
