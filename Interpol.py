from scipy.interpolate import Akima1DInterpolator
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Определение функций ---

def constrained_gaussian(x, A, sigma, mu_fixed):
    return A * np.exp(-(x - mu_fixed) ** 2 / (2 * sigma ** 2))

def bockasten_abel_transformation(N, coefficients, r_max):
    e = coefficients.T @ N
    return e / (r_max * 0.001)

def fit_and_interpolate(radii, intensities, fine_points=40, mu_fixed=None):
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    intensities = intensities[sorted_indices]

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

def plot_data(x, y, x_fit=None, y_fit=None, title="", xlabel="", ylabel=""):
    size = {'size' : 14}
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'o', label="Experimental data", color='blue')
    if x_fit is not None and y_fit is not None:
        plt.plot(x_fit, y_fit, color='green', label="Gaussian Fit")
    plt.xlabel(xlabel, fontdict=size)
    plt.ylabel(ylabel, fontdict=size)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Матрица коэффициентов Боккастена
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

    abel_results = pd.DataFrame()
    
    # --- ЦИКЛ ОБРАБОТКИ ФАЙЛОВ ---
    for j in range(0, 3):
        name_line = input(f'Enter line wavelength for iteration {j+1}: ').strip()
        
        file_path = Path('results') / f'{name_line}_gauss.xlsx'
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            continue

        data = pd.read_excel(file_path, header=None)
        radius_columns = data.iloc[:, ::2]
        intensity_columns = data.iloc[:, 1::2]

        for i, column in enumerate(intensity_columns.columns):
            radii_all = radius_columns.iloc[:, i].dropna().values
            intensities_all = intensity_columns.iloc[:, i].dropna().values

            if len(np.unique(radii_all)) != len(radii_all):
                uniq_r, idx = np.unique(radii_all, return_index=True)
                radii_all = uniq_r
                intensities_all = intensities_all[idx]

            sorted_idx = np.argsort(radii_all)
            radii_all = radii_all[sorted_idx]
            intensities_all = intensities_all[sorted_idx]

            mu_fixed_global = 0.0
            popt_global, mu_fixed_global, fine_r_all, smoothed_all = fit_and_interpolate(
                radii_all, intensities_all, fine_points=80, mu_fixed=mu_fixed_global
            )
            if popt_global is None:
                continue
            
            # Предварительный просмотр гаусса
            plot_data(radii_all, intensities_all, fine_r_all, smoothed_all,
                      title=f'Gaussian Fit: {name_line} Line {i+1}',
                      xlabel='r, mm', ylabel='I, W/m²')

            mask_left = radii_all < 0
            mask_right = radii_all > 0
            abel_branches = {}

            for branch_name, mask in (("left", mask_left), ("right", mask_right)):
                if not np.any(mask): continue

                r_branch = np.abs(radii_all[mask])
                if not np.any(np.isclose(r_branch, 0.0, atol=1e-9)):
                    r_branch = np.concatenate(([0.0], r_branch))

                r_branch = np.unique(r_branch)
                r_branch.sort()

                matched_intensities = constrained_gaussian(r_branch, *popt_global, mu_fixed_global)

                n = len(matched_intensities)
                max_n = bockasten_coefficients.shape[0]
                if n > max_n:
                    n = max_n
                    r_branch = r_branch[:n]
                    matched_intensities = matched_intensities[:n]

                r_max = r_branch.max()
                if r_max == 0: continue

                C = bockasten_coefficients[:n, :n]
                abelized_intensities = bockasten_abel_transformation(matched_intensities, C, r_max)

                abel_branches[branch_name] = (r_branch, abelized_intensities)

                col_prefix = f'{name_line}_{branch_name}_L{i+1}'
                abel_block = pd.DataFrame({
                    f'{col_prefix}_R': pd.Series(r_branch),
                    f'{col_prefix}_I': pd.Series(abelized_intensities),
                })
                abel_results = pd.concat([abel_results, abel_block], axis=1)

    print("Saving Abel results to abel_results.xlsx...")
    abel_results.to_excel('abel_results.xlsx', index=False)
    
    # --- БЛОК 1: Сводный график (Raw Abel) ---
    print("Plotting cumulative epsilon vs radius (Raw)...")
    plt.figure(figsize=(12, 8))
    pairs_plotted = 0
    num_cols = abel_results.shape[1]
    
    for k in range(0, num_cols, 2):
        if k+1 >= num_cols: break
        r = abel_results.iloc[:, k]
        e = abel_results.iloc[:, k+1]
        valid_idx = ~np.isnan(r) & ~np.isnan(e)
        r_clean = r[valid_idx].values
        e_clean = e[valid_idx].values
        
        if len(r_clean) > 0:
            label_name = abel_results.columns[k+1].replace('_I', '')
            plt.plot(r_clean, e_clean, marker='o', label=label_name)
            pairs_plotted += 1

    size_font = {'size' : 14}
    plt.xlabel("r, mm", fontdict=size_font)
    plt.ylabel("ε(r), W / m³", fontdict=size_font)
    plt.xlim(0, 4)
    plt.grid(True)
    plt.title("Combined ε(r) (Raw Data)")
    if pairs_plotted > 0:
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("epsilon_vs_radius_lines.png", dpi=200)
    plt.show()

    # --- БЛОК 2: Выравнивание (Alignment) ---
    print("Starting alignment (trimming to min radius)...")
    
    if abel_results.empty:
        logging.warning("No Abel results to align.")
        return

    max_radii_left = []
    max_radii_right = []

    for k in range(0, num_cols, 2):
        col_name_r = abel_results.columns[k]
        r_vals = abel_results.iloc[:, k].dropna().values
        if len(r_vals) == 0: continue
        current_max = np.max(r_vals)
        if "_left_" in col_name_r:
            max_radii_left.append(current_max)
        elif "_right_" in col_name_r:
            max_radii_right.append(current_max)

    limit_left = min(max_radii_left) if max_radii_left else 0
    limit_right = min(max_radii_right) if max_radii_right else 0

    print(f"Alignment limit Left: {limit_left:.4f} mm")
    print(f"Alignment limit Right: {limit_right:.4f} mm")

    grid_points = 100
    common_grid_left = np.linspace(0, limit_left, grid_points) if limit_left > 0 else []
    common_grid_right = np.linspace(0, limit_right, grid_points) if limit_right > 0 else []

    aligned_left_df = pd.DataFrame()
    aligned_right_df = pd.DataFrame()

    if len(common_grid_left) > 0:
        aligned_left_df['Radius_Common_Left'] = common_grid_left
    if len(common_grid_right) > 0:
        aligned_right_df['Radius_Common_Right'] = common_grid_right

    for k in range(0, num_cols, 2):
        col_name_r = abel_results.columns[k]
        col_name_i = abel_results.columns[k+1]
        
        df_temp = abel_results[[col_name_r, col_name_i]].dropna()
        if df_temp.empty: continue
        
        r_vals = df_temp[col_name_r].values
        i_vals = df_temp[col_name_i].values
        
        sort_idx = np.argsort(r_vals)
        r_vals = r_vals[sort_idx]
        i_vals = i_vals[sort_idx]
        if len(r_vals) < 2: continue

        interpolator = Akima1DInterpolator(r_vals, i_vals)
        new_col_name = col_name_i.replace('_I', '')

        if "_left_" in col_name_r and len(common_grid_left) > 0:
            interp_vals = interpolator(common_grid_left)
            aligned_left_df[new_col_name] = interp_vals
        elif "_right_" in col_name_r and len(common_grid_right) > 0:
            interp_vals = interpolator(common_grid_right)
            aligned_right_df[new_col_name] = interp_vals

    aligned_final = pd.concat([aligned_left_df, aligned_right_df], axis=1)
    aligned_final.to_excel('aligned_results.xlsx', index=False)
    print("Aligned data saved to aligned_results.xlsx")

    print("Plotting aligned data...")
    plt.figure(figsize=(12, 8))
    
    
    # Рисуем левые ветки (Left -> Negative)
    if 'Radius_Common_Left' in aligned_final.columns:
        r_l = aligned_final['Radius_Common_Left']
        for col in aligned_left_df.columns:
            if col == 'Radius_Common_Left': continue
            # Укоротим имя для легенды
            clean_label = col.replace(name_line, '').replace('_left_', ' Left ').strip(' _')
            plt.plot(-r_l, aligned_left_df[col], linestyle='--', linewidth=2, label=f"{clean_label}")

    # Рисуем правые ветки (Right -> Positive)
    if 'Radius_Common_Right' in aligned_final.columns:
        r_r = aligned_final['Radius_Common_Right']
        for col in aligned_right_df.columns:
            if col == 'Radius_Common_Right': continue
            clean_label = col.replace(name_line, '').replace('_right_', ' Right ').strip(' _')
            plt.plot(r_r, aligned_right_df[col], linewidth=2, label=f"{clean_label}")
            
    # Применение стилей
    size_font = {'size' : 14}
    plt.xlabel("r, mm", fontdict=size_font)
    plt.ylabel("ε(r), W / m³ (Aligned)", fontdict=size_font)
    plt.title("Aligned Radial Distribution of Emissivity")
    
    # Лимит ставим чуть шире максимума
    max_lim = max(limit_left, limit_right)
    plt.xlim(-(max_lim + 0.5), (max_lim + 0.5))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    out_path_aligned = "aligned_epsilon_profile.png"
    plt.savefig(out_path_aligned, dpi=200)
    print(f"Aligned plot saved to {out_path_aligned}")
    plt.show()

if __name__ == "__main__":
    main()