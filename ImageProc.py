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

def find_cutoff_point(data, start_index: int, threshold: float) -> int:
    for i in range(start_index, len(data)):
        if data[i] < threshold:
            return i
    return len(data) - 1

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian(x_values, y_values):
    gauss_mod = Model(gaussian)
    params = gauss_mod.make_params(A=np.max(y_values), mu=x_values[np.argmax(y_values)], sigma=0.1)
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
    window_length = int(input('Enter smoothing window length (odd number, e.g., 75): ') or 75)
    polyorder = int(input('Enter polynomial order for smoothing (e.g., 3): ') or 3)

    bottom_wl = float(input('Enter bottom wavelength: '))
    top_wl = float(input('Enter top wavelength: '))



    file_path = Path(file_base + '.xlsx')
    data = load_data(file_path)

    data = cut_line(data, bottom_wl, top_wl)
    print(f"Cutted file length {len(data)}")

    wavelengths = data.iloc[:, 0].values
    intensities = data.iloc[:, 1:]
    x = np.linspace(0, intensities.shape[1] - 1, intensities.shape[1])

    smoothed = smooth_data(intensities, window_length, polyorder)
    top_lines = find_top_lines(smoothed)
    sum_top = top_lines.sum(axis=1)

    # График топ-5 линий и суммы
    plt.figure(figsize=(12, 8))
    for line in top_lines.columns:
        plt.plot(x, top_lines[line], label=f"{line}")
    plt.plot(x, sum_top, label="Sum of Top 5 Lines", linestyle="--", color="black")
    plt.xlabel("Pixels")
    plt.ylabel("Intensity")
    plt.title("Top 5 Spectral Lines and Summed Intensity")
    plt.legend()
    plt.grid()
    plt.show()
    threshold = float(input('Enter threshold: '))
    print('Enter max position manually? (Enter 1 if NO)')
    manual_choice = int(input().strip())
    if manual_choice == 1:
        max_pos = sum_top.idxmax()
    else:
        max_pos = manual_choice

    cutoff_idx = find_cutoff_point(sum_top, max_pos, threshold)

    pixel_positions = [(i - max_pos) * 0.0155 for i in range(len(sum_top))]
    num_points = 10
    right_indices = np.linspace(max_pos, cutoff_idx, num_points, dtype=int)
    right_positions = [pixel_positions[i] for i in right_indices]

    # График правой ветки
    plt.figure(figsize=(10, 6))
    plt.plot(pixel_positions, sum_top, label="Sum of Top 5 Lines", color='blue')
    plt.scatter([pixel_positions[i] for i in right_indices], [sum_top[i] for i in right_indices], color='red', zorder=5, label="Right Branch Points")
    plt.axvline(0, color='green', linestyle='--', label="Center")
    plt.xlabel("Pixel Position (Step = 0.0155)")
    plt.ylabel("Intensity")
    plt.title("Right Branch of Summed Top 5 Lines")
    plt.legend()
    plt.grid()
    plt.show()

    results = pd.DataFrame()
    for idx in range(len(intensities)):
        smoothed_line = smoothed[f"Line_{idx+1}"].values
        results[idx+1] = [smoothed_line[i] for i in right_indices]

    results.loc[-1] = list(wavelengths)
    results.index = results.index + 1
    results.sort_index(inplace=True)

    (Path.cwd() / 'results').mkdir(exist_ok=True)
    results.transpose().to_excel(Path('results') / 'res_bef_gauss.xlsx', index=False)

    fit_results = []
    x_vals = results.iloc[0, 1:].values
    x_dense = np.linspace(min(x_vals), max(x_vals), 5000)

    for i in range(1, len(results)):
        y_vals = results.iloc[i, 1:].values

        if np.max(y_vals) < 1e-3:
            logging.info(f"Skipping row {i}: low signal.")
            continue

        fit = fit_gaussian(x_vals, y_vals)
        if fit:
            area = fit['A'] * fit['Sigma'] * np.sqrt(2 * np.pi)
            fit_results.append({'Radius': right_positions[i-1], name_line: area})

            # График фиттинга
            plt.figure(figsize=(10, 6))
            plt.scatter(x_vals, y_vals, label='Data')
            plt.plot(x_dense, gaussian(x_dense, fit['A'], fit['Mu'], fit['Sigma']), label='Fit', color='red')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.title(f'Gaussian Fit - Row {i}')
            plt.legend()
            plt.grid()
        else:
            logging.warning(f"Fit failed for row {i}.")

    plt.show()

    fit_df = pd.DataFrame(fit_results)
    fit_df.to_excel(Path('results') / f'{name_line}_gauss.xlsx', index=False)

if __name__ == "__main__":
    main()
