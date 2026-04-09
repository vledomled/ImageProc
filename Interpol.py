from scipy.interpolate import Akima1DInterpolator, UnivariateSpline
from scipy.optimize import curve_fit
from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.figsize": (7, 6),
    "figure.dpi": 150,
    "figure.autolayout": True
})

# --- Определение функций ---

def gaussian_model(r, a, sigma):
    return a * np.exp(-(r**2) / (2 * sigma**2))

def generate_bockasten_matrix(n):
    """Генерация матрицы коэффициентов a_jk по точным формулам Бокастена."""
    a = np.zeros((n, n))
    
    def eval_int(p, lower, upper, j_val):
        def F(x):
            if j_val == 0:
                if p == 2: return 0.5 * x**2
                if p == 1: return x
                if p == 0: return math.log(x)
            else:
                if p == 2: return 0.5 * x * math.sqrt(x**2 - j_val**2) + 0.5 * j_val**2 * math.log(x + math.sqrt(x**2 - j_val**2))
                if p == 1: return math.sqrt(x**2 - j_val**2)
                if p == 0: return math.log(x + math.sqrt(x**2 - j_val**2))
        return F(upper) - F(lower)

    for m in range(n):
        if m == 0:
            coeffs = {0: [0, -3.5, 2.25], 1: [0, 4, -3], 2: [0, -0.5, 0.75]}
        elif m == n - 1:
            coeffs = {n-2: [0.5 - n, 1, 0], n-1: [2*n - 2, -2, 0]}
        else:
            coeffs = {
                m-1: [-0.5*m**2 - m - 1/3, m+1, -0.5],
                m:   [1.5*m**2 + 2*m - 0.5, -(3*m+2), 1.5],
                m+1: [-1.5*m**2 - m + 1, 3*m+1, -1.5],
                m+2: [0.5*m**2 - 1/6, -m, 0.5]
            }
            coeffs = {k: v for k, v in coeffs.items() if k < n}
                
        for k, c_arr in coeffs.items():
            c0, c1, c2 = c_arr
            for j in range(n):
                if m < j: continue
                val = c2 * eval_int(2, m, m+1, j) + c1 * eval_int(1, m, m+1, j)
                if not (j == 0 and m == 0 and c0 == 0):
                    val += c0 * eval_int(0, m, m+1, j)
                a[j, k] += (-n / math.pi) * val
    return a

def bockasten_abel_transformation(N, coefficients, r_max):
    e = coefficients @ N
    return e / (r_max * 0.001)

def main():
    file_path = Path('results') / 'All_spectra_areas.xlsx'
    if not file_path.exists():
        file_path = Path('All_spectra_areas.xlsx')
        if not file_path.exists():
            logging.error(f"Файл данных не найден: {file_path}")
            return

    data = pd.read_excel(file_path)
    lines = data['Line'].unique()
    (Path.cwd() / 'results').mkdir(exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    color_map = {line: colors[idx % len(colors)] for idx, line in enumerate(lines)}

    # Подготовка к хранению результатов
    abel_results = pd.DataFrame()
    raw_plots_data = []
    
    # Создаем холст для графика сглаживания (контрольный)
    plt.figure(figsize=(8, 6))
    plt.title("Gaussian Smoothing Quality Check")

    print("\n--- Выполнение аппроксимации Гауссом и преобразования Абеля ---")
    for line_name in lines:
        line_data = data[data['Line'] == line_name]
        radii_all = line_data['Radius_mm'].values
        intensities_all = line_data['Area'].values

        mask_left = radii_all <= 0
        mask_right = radii_all >= 0

        for branch_name, mask in (("left", mask_left), ("right", mask_right)):
            branch_df = pd.DataFrame({
                'R_abs': np.abs(radii_all[mask]),
                'I': intensities_all[mask]
            }).sort_values('R_abs')

            r_branch = branch_df['R_abs'].values
            i_branch = branch_df['I'].values
            n_points = len(r_branch)

            if n_points < 5: # Для Гаусса нужно меньше точек, чем для сплайна
                logging.warning(f"Мало точек для {line_name} ({branch_name})")
                continue

            r_max = r_branch.max()
            if r_max == 0: continue

            # --- ГАУССОВА АППРОКСИМАЦИЯ ---
            try:
                # Начальные параметры: амплитуда и ширина
                p0 = [i_branch.max(), r_max / 2]
                popt, _ = curve_fit(gaussian_model, r_branch, i_branch, p0=p0)
                
                # Создаем гладкую кривую
                i_branch_smooth = gaussian_model(r_branch, *popt)
                
                # Отрисовка на графике сглаживания
                r_plot_smooth = -r_branch if branch_name == 'left' else r_branch
                plt.scatter(r_plot_smooth, i_branch, color=color_map[line_name], alpha=0.3, s=20)
                plt.plot(r_plot_smooth, i_branch_smooth, color=color_map[line_name], 
                         linestyle='--', label=f"{line_name} {branch_name}" if branch_name == 'right' else "")
                
            except Exception as e:
                logging.error(f"Ошибка аппроксимации {line_name}: {e}")
                i_branch_smooth = i_branch # Если не вышло, берем как есть

            # --- МАТРИЦА БОКАСТЕНА ---
            bockasten_coefficients = generate_bockasten_matrix(n_points)
            abelized_intensities = bockasten_abel_transformation(i_branch_smooth, bockasten_coefficients, r_max)

            # Сохранение
            col_prefix = f'{line_name}_{branch_name}'
            abel_block = pd.DataFrame({
                f'{col_prefix}_R': pd.Series(r_branch),
                f'{col_prefix}_Eps': pd.Series(abelized_intensities),
                f'{col_prefix}_I_raw': pd.Series(i_branch),
                f'{col_prefix}_I_smooth': pd.Series(i_branch_smooth)
            })
            abel_results = pd.concat([abel_results, abel_block], axis=1)
            raw_plots_data.append((line_name, branch_name, r_branch, abelized_intensities))

    plt.xlabel("Radius $r$ [mm]")
    plt.ylabel("Intensity [a.u.]")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=9, loc='upper right')
    plt.savefig(Path('results') / "smoothing_check.png")
    plt.show()

    # --- Далее идет твой блок построения ε(r) и Aligned графиков без изменений ---
    # (Оставил логику сохранения в Excel и финальные графики как в твоем исходнике)
    out_abel = Path('results') / 'abel_results_gaussian.xlsx'
    abel_results.to_excel(out_abel, index=False)
    print(f"\nПрофили ε(r) сохранены в {out_abel}")

    print("Построение профилей ε(r)...")
    plt.figure()
    
    for line_name, branch_name, r, eps in raw_plots_data:
        r_plot = -r if branch_name == 'left' else r
        label = line_name if branch_name == 'right' else "_nolegend_"
        plt.plot(r_plot, eps, color=color_map[line_name], lw=2, label=label)

    plt.xlabel("Radius $r$ [mm]")
    plt.ylabel("Emissivity $\\varepsilon(r)$ [W/m$^3$]")
    plt.title("Radial Emissivity Profiles (Abel Inverted)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig(Path('results') / "epsilon_vs_radius.png")
    plt.show()

    limit_left = min([np.max(r) for _, b, r, _ in raw_plots_data if b == 'left'], default=0)
    limit_right = min([np.max(r) for _, b, r, _ in raw_plots_data if b == 'right'], default=0)

    print(f"\nГраницы выравнивания (по минимальному размеру): Left {limit_left:.3f} mm, Right {limit_right:.3f} mm")

    grid_points = 100
    common_grid_left = np.linspace(0, limit_left, grid_points) if limit_left > 0 else []
    common_grid_right = np.linspace(0, limit_right, grid_points) if limit_right > 0 else []

    aligned_left_df = pd.DataFrame({'Radius_mm': common_grid_left}) if limit_left > 0 else pd.DataFrame()
    aligned_right_df = pd.DataFrame({'Radius_mm': common_grid_right}) if limit_right > 0 else pd.DataFrame()

    for line_name, branch_name, r, eps in raw_plots_data:
        interpolator = Akima1DInterpolator(r, eps)
        
        if branch_name == 'left' and limit_left > 0:
            aligned_left_df[line_name] = interpolator(common_grid_left)
        elif branch_name == 'right' and limit_right > 0:
            aligned_right_df[line_name] = interpolator(common_grid_right)

    aligned_final = pd.concat([aligned_left_df.add_suffix('_Left'), aligned_right_df.add_suffix('_Right')], axis=1)
    out_aligned = Path('results') / 'aligned_results.xlsx'
    aligned_final.to_excel(out_aligned, index=False)
    
    print("Построение выровненных графиков...")
    plt.figure()
    
    for line_name in lines:
        col_color = color_map[line_name]
        
        if f"{line_name}_Left" in aligned_final.columns:
            r_l = aligned_final['Radius_mm_Left']
            e_l = aligned_final[f"{line_name}_Left"]
            plt.plot(-r_l, e_l, linestyle='--', color=col_color, lw=2, label=f"{line_name} (Left)")
            
        if f"{line_name}_Right" in aligned_final.columns:
            r_r = aligned_final['Radius_mm_Right']
            e_r = aligned_final[f"{line_name}_Right"]
            plt.plot(r_r, e_r, linestyle='-', color=col_color, lw=2, label=f"{line_name} (Right)")

    plt.xlabel("Radius $r$ [mm]")
    plt.ylabel("Aligned $\\varepsilon(r)$ [W/m$^3$]")
    plt.title("Aligned Radial Emissivity Profiles")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(Path('results') / "aligned_epsilon.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()